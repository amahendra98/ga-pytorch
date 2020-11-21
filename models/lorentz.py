import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch.autograd import Variable, grad
import time
from torch.nn.parameter import Parameter as P
from torch import pow, add, mul, div
import numpy as np
import random


class lorentz_model(nn.Module):
    def __init__(self, linear):
        super(lorentz_model,self).__init__()
        self.linear = linear[0:-1]
        self.num_spec_point = linear[-1]
        self.num_lorentz_osc = 1
        self.freq_low = 0.5
        self.freq_high = 5
        self.device = None

        self.linears = nn.ModuleList([])
        #self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(self.linear[0:-1]):  # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, self.linear[ind + 1], bias=True))
            #self.bn_linears.append(nn.BatchNorm1d(self.linear[ind + 1], track_running_stats=True, affine=True))

        self.lin_w0 = nn.Linear(self.linear[-1], self.num_lorentz_osc, bias=False)
        self.lin_wp = nn.Linear(self.linear[-1], self.num_lorentz_osc, bias=False)
        self.lin_g = nn.Linear(self.linear[-1], self.num_lorentz_osc, bias=False)
        torch.nn.init.uniform_(self.lin_g.weight, a=0.0, b=0.1)

    def forward(self, G):
        out = G
        for cnxn in self.linears:
            out = F.relu(cnxn(out))

        w0 = F.relu(self.lin_w0(out))
        wp = F.relu(self.lin_wp(out))
        g = F.relu(self.lin_g(out))

        w0_out = w0
        wp_out = wp
        g_out = g

        w0 = w0.unsqueeze(2) * 1  # size -> [1024,4,1]
        wp = wp.unsqueeze(2) * 1
        g = g.unsqueeze(2) * 0.1

        w0 = w0.expand(out.size(0), self.lin_g.weight.size(0), 300)  # size -> [1024,4,300]
        wp = wp.expand_as(w0)
        g = g.expand_as(w0)
        w = torch.arange(0.5, 5, (5 - 0.5) / 300, device=self.device)
        w_expand = w.expand_as(g)

        e2 = div(mul(pow(wp, 2), mul(w_expand, g)),
        add(pow(add(pow(w0, 2), -pow(w_expand, 2)), 2), mul(pow(w_expand, 2), pow(g, 2))))

        e2 = torch.sum(e2, 1)
        T = e2.float()
        return T, (w0_out,wp_out,g_out) #For just one lorentzian

    def collect_mutations(self, mut, num, dev):
        RAND = []
        for p in self.parameters():
            s = p.size()
            if len(s) == 1:
                RAND.append(torch.mul(torch.randn(num,s[0],requires_grad=False,device=self.device), mut))
            if len(s) == 2:
                RAND.append(torch.mul(torch.randn(num,s[0],s[1],requires_grad=False, device=self.device), mut))
        return RAND

    def set_device(self, dev):
        self.device = dev

    @staticmethod
    def bc_func(z): #For single lorentzian only
        l = []
        for mp in z.parameters():
            l.append(mp)
        return torch.cat((l[0],l[1].unsqueeze(1),l[2],l[3].unsqueeze(1),l[4][0].unsqueeze(1),l[5][0].unsqueeze(1),
                          l[6][0].unsqueeze(1)), 1)

    def p_list(self):
        return lorentz_model.bc_func(self).cpu().numpy()

    #From Safe Mutations Code
    #function to return current pytorch gradient in same order as genome's flattened parameter vector
    def extract_grad(self):
        tot_size = self.count_parameters()
        #pvec = np.zeros(tot_size, np.float32)

        if self.linears[0].weight.is_cuda:
            pvec = torch.zeros(tot_size,dtype=torch.float32,device=self.device)
        else:
            pvec = torch.zeros(tot_size, dtype=torch.float32)

        count = 0
        for param in self.parameters():
            sz = torch.flatten(param.grad.data).shape[0]
            #sz = param.grad.data.numpy().flatten().shape[0]
            #pvec[count:count + sz] = param.grad.data.numpy().flatten()
            pvec[count:count + sz] = torch.flatten(param.grad.data)
            count += sz
        #return pvec.copy()
        return pvec.clone()

    #function to grab current flattened neural network weights
    def extract_parameters(self):
        tot_size = self.count_parameters()
        #pvec = np.zeros(tot_size, np.float32)
        if self.linears[0].weight.is_cuda:
            pvec = torch.zeros(tot_size,dtype=torch.float32,device=self.device)
        else:
            pvec = torch.zeros(tot_size, dtype=torch.float32)
        count = 0
        for param in self.parameters():
            sz = torch.flatten(param.data).shape[0]
            #sz = param.data.numpy().flatten().shape[0]
            pvec[count:count+sz] = torch.flatten(param.data)
            #pvec[count:count + sz] = param.data.numpy().flatten()
            count += sz
        #return pvec.copy()
        return torch.clone(pvec)

    #function to inject a flat vector of ANN parameters into the model's current neural network weights
    def inject_parameters(self, pvec):
        tot_size = self.count_parameters()
        count = 0

        for param in self.parameters():
            sz = torch.flatten(param.data).shape[0]
            #sz = param.data.numpy().flatten().shape[0]
            raw = pvec[count:count + sz]
            reshaped = torch.reshape(raw,param.data.shape)
            #reshaped = raw.reshape(param.data.numpy().shape)
            param.data = reshaped
            #param.data = torch.from_numpy(reshaped)
            count += sz

        #print(pvec)
        return pvec

    #count how many parameters are in the model
    def count_parameters(self):
        count = 0
        for param in self.parameters():
            #print param.data.numpy().shape
            count += torch.flatten(param.data).shape[0]
            #count += param.data.cpu().numpy().flatten().shape[0]
        return count


    # TODO: Fix up Mutate function to reflect non-individual based format and other modifications
    def mutate(self,mag,states, mutation='regular'):
        with torch.enable_grad():
            #plain mutation is normal ES-style mutation
            if mutation=='regular':
                self.mutate_plain(mag)
            elif mutation.count("SM-G")>0:
                #smog_target is target-based smog where we aim to perturb outputs
                self.mutate_sm_g(mutation,states,mag)

            #smog_grad is TRPO-based smog where we attempt to induce limited policy change
        # elif mutation.count("SM-R")>0:
        #     self.mutate_sm_r(
        #             self.genome,
        #             individual.global_model,
        #             individual.env,
        #             states=self.states,
        #             **kwargs)
            else:
                assert False


    def mutate_plain(self,mag=0.05, **kwargs):
        # do_policy_check = False
        params = self.extract_parameters()
        print(mag)
        delta = torch.randn(*params.shape, dtype=torch.float32, device=self.device) * mag
        new_params = params + delta

        #diff = np.sqrt(((new_params - params) ** 2).sum())
        diff = torch.sqrt(torch.sum((new_params - params) ** 2))

        print("Divergence: ", diff)

        # if do_policy_check:
        #     output_dist = check_policy_change(params, new_params, kwargs['states'])
        #     print("mutation size: ", diff, "output distribution change:", output_dist)
        # else:
        #     print("mutation size: ", diff)

        # return new_params
        self.inject_parameters(new_params)

    def mutate_sm_r(self,
                    params,
                    #model,
                    # env,
                    # verbose=True,
                    states=None,
                    power=0.01):
        # global state_archive

        #model.inject_parameters(params.copy())
        self.inject_parameters(params.copy())

        # if states == None:
        #     states = state_archive

        # delta = np.random.randn(*(params.shape)).astype(np.float32)
        delta = torch.mul(torch.randn(*(params.shape), requires_grad=False, device=self.device), power)
        # delta = delta / np.sqrt((delta ** 2).sum())
        delta = delta / torch.sqrt(torch.sum(delta ** 2))

        # sz = min(100, len(state_archive))
        #np_obs = np.array(random.sample(state_archive, sz), dtype=np.float32)
        #verification_states = torch.Variable(torch.from_numpy(np_obs), requires_grad=False)

        ' Pass verification states through policy to get output = old policy'
        verification_states = Variable(states, requires_grad=False)
        old_policy = self.__call__(verification_states)
        # old_policy = model(verification_states)
        #old_policy = output.data.numpy()

        # do line search
        threshold = power
        search_rounds = 15

        def search_error(x, raw=False):
            new_params = params + delta * x
            self.inject_parameters(new_params)

            #output = model(verification_states).data.numpy()
            output = self.__call__(verification_states)

            #change = ((output - old_policy) ** 2).mean()
            change = torch.mean((output - old_policy) ** 2)
            if raw:
                return change
            return (change - threshold) ** 2

        mult = minimize_scalar(search_error, tol=0.01 ** 2, options={'maxiter': search_rounds, 'disp': True})
        new_params = params + delta * mult.x

        print
        'Distribution shift:', search_error(mult.x, raw=True)
        print
        "SM-R scaling factor:", mult.x
        # diff = np.sqrt(((new_params - params) ** 2).sum())
        # print("mutation size: ", diff)
        diff = torch.sqrt(torch.sum((new_params - params) ** 2))
        print("mutation size: Are these dimensions righ? ", diff)
        return new_params

    # def check_policy_change(p1, p2, states):
    #     model.inject_parameters(p1.copy())
    #     sz = min(100, len(states))
    #
    #     verification_states = np.array(random.sample(states, sz), dtype=np.float32)
    #     verification_states = Variable(torch.from_numpy(verification_states), requires_grad=False)
    #     old_policy = model(verification_states).data.numpy()
    #     old_policy = Variable(torch.from_numpy(old_policy), requires_grad=False)
    #
    #     model.inject_parameters(p2.copy())
    #     model.zero_grad()
    #     new_policy = model(verification_states)
    #     divergence_loss_fn = torch.nn.MSELoss(size_average=True)
    #     divergence_loss = divergence_loss_fn(new_policy, old_policy)
    #
    #     return divergence_loss.data[0]

    def mutate_sm_g(self,mutation,
                    states,
                    power=0.05,
                    **kwargs):

        # global state_archive

        # model.inject_parameters(params.copy())
        params = self.extract_parameters()

        # if no states passed in, use global state archive
        # if states == None:
            # states = state_archive

        # sub-sample experiences from parent
        # sz = min(100, len(states))
        sz = len(states)
        # verification_states = np.array(random.sample(states, sz), dtype=np.float32)
        # verification_states = torch.Variable(torch.from_numpy(verification_states), requires_grad=False)

        verification_states = Variable(states, requires_grad=False)
        #old_policy = model(verification_states)
        old_policy = self.__call__(verification_states)[0]

        # run experiences through model
        # NOTE: for efficiency, could cache these during actual evalution instead of recalculating
        # old_policy = model(verification_states)

        num_outputs = old_policy.size()[1]

        abs_gradient = False
        avg_over_time = False
        second_order = False

        if mutation.count("ABS") > 0:
            abs_gradient = True
            avg_over_time = True
        if mutation.count("SO") > 0:
            second_order = True

        # generate normally-distributed perturbation
        #delta = np.random.randn(*params.shape).astype(np.float32) * power
        delta = torch.mul(torch.randn(*(params.shape), requires_grad=False,device=self.device), power)

        if second_order:
            print
            'SM-G-SO'
            #np_copy = np.array(old_policy.data.numpy(), dtype=np.float32)
            #_old_policy_cached = torch.Variable(torch.from_numpy(np_copy), requires_grad=False)
            #loss = ((old_policy - _old_policy_cached) ** 2).sum(1).mean(0)
            copy = torch.array(old_policy.data, dtype=torch.float32)
            _old_policy_cached = Variable(copy, requires_grad=False)
            loss = torch.mean(torch.sum((old_policy - _old_policy_cached) ** 2, 1), 0)
            # loss_gradient = grad(loss, self.parameters(), create_graph=True)
            loss_gradient = grad(loss, self.parameters(), create_graph=True)
            flat_gradient = torch.cat([grads.view(-1) for grads in loss_gradient])  # .sum()

            #direction = (delta / np.sqrt((delta ** 2).sum()))
            #direction_t = torch.Variable(torch.from_numpy(direction), requires_grad=False)
            #grad_v_prod = (flat_gradient * direction_t).sum()
            direction = (delta / torch.sqrt(torch.sum(delta ** 2)))
            direction_t = Variable(direction, requires_grad=False)
            grad_v_prod = torch.sum(flat_gradient * direction_t)
            second_deriv = torch.autograd.grad(grad_v_prod, self.parameters())
            sensitivity = torch.cat([g.contiguous().view(-1) for g in second_deriv])
            scaling = torch.sqrt(torch.abs(sensitivity).data)

        elif not abs_gradient:
            #print
            "SM-G-SUM"
            tot_size = self.count_parameters()
            jacobian = torch.zeros(num_outputs, tot_size, device=self.device)
            grad_output = torch.zeros(*old_policy.size(), device=self.device)

            for i in range(num_outputs):
                self.zero_grad()
                grad_output.zero_()
                grad_output[:, i] = 1.0
                old_policy.backward(grad_output, retain_graph=True) #HAD RETAIN_VARIABLES! pytorch forum says retain_graph is newer version of it
                jacobian[i] = self.extract_grad()

            #scaling = torch.sqrt((jacobian ** 2).sum(0))
            scaling = torch.sqrt(torch.sum(jacobian ** 2,0))

        else:
            print
            "SM-G-ABS"
            # NOTE: Expensive because quantity doesn't slot naturally into TF/pytorch framework
            tot_size = self.count_parameters()
            jacobian = torch.zeros(num_outputs, tot_size, sz)
            grad_output = torch.zeros([1, num_outputs])  # *old_policy.size())

            for i in range(num_outputs):
                for j in range(sz):
                    #old_policy_j = model(verification_states[j:j + 1])
                    #model.zero_grad()
                    old_policy_j = self.__call__(verification_states[j:j + 1])
                    self.zero_grad()
                    grad_output.zero_()

                    grad_output[0, i] = 1.0

                    old_policy_j.backward(grad_output, retain_variables=True)
                    #jacobian[i, :, j] = torch.from_numpy(model.extract_grad())
                    jacobian[i, :, j] = self.extract_grad()

            mean_abs_jacobian = torch.abs(jacobian).mean(2)
            scaling = torch.sqrt((mean_abs_jacobian ** 2).sum(0))

        #scaling = scaling.numpy()

        # Avoid divide by zero error
        # (intuition: don't change parameter if it doesn't matter)
        scaling[scaling == 0] = 1.0

        # Avoid straying too far from first-order approx
        # (intuition: don't let scaling factor become too enormous)
        scaling[scaling < 0.01] = 0.01

        # rescale perturbation on a per-weight basis
        delta /= scaling

        # generate new perturbation
        new_params = params + delta

        #model.inject_parameters(new_params)
        self.inject_parameters(new_params)
        #old_policy = old_policy.data.numpy()

        # restrict how far any dimension can vary in one mutational step
        weight_clip = 0.2

        # currently unused: SM-G-*+R (using linesearch to fine-tune)
        mult = 0.05

        if mutation.count("R") > 0:
            linesearch = True
            threshold = power
        else:
            linesearch = False

        if linesearch == False:
            search_rounds = 0
        else:
            search_rounds = 15

        def search_error(x, raw=False):
            final_delta = delta * x
            final_delta = torch.clip(final_delta,-weight_clip,weight_clip)
            #final_delta = np.clip(final_delta, -weight_clip, weight_clip)
            new_params = params + final_delta
            #model.inject_parameters(new_params)
            self.inject_parameters(new_params)

            #output = model(verification_states).data.numpy()
            output = self.__call__(verification_states)
            change = torch.sqrt(((output - old_policy) ** 2).sum(1)).mean()
            #change = np.sqrt(((output - old_policy) ** 2).sum(1)).mean()

            if raw:
                return change
            return (change - threshold) ** 2

        if linesearch:
            mult = minimize_scalar(search_error, bounds=(0, 0.1, 3), tol=(threshold / 4) ** 2,
                                   options={'maxiter': search_rounds, 'disp': True})
            print
            "linesearch result:", mult
            chg_amt = mult.x
        else:
            # if not doing linesearch
            # don't change perturbation
            chg_amt = 1.0

        final_delta = delta * chg_amt
        #print
        #'perturbation max magnitude:', final_delta.max()

        #final_delta = np.clip(delta, -weight_clip, weight_clip)
        final_delta = torch.clamp(delta, -weight_clip, weight_clip)
        new_params = params + final_delta

        #print
        #'max post-perturbation weight magnitude:', abs(new_params).max()

        #if verbose:
        #    print("divergence:", search_error(chg_amt, raw=True))

        #diff = np.sqrt(((new_params - params) ** 2).sum())
        #diff = torch.sqrt(torch.sum((new_params - params) ** 2))
        #print("mutation size: ", diff)

        #return new_params
        self.inject_parameters(new_params)


if __name__ == '__main__':
    m1 = lorentz_model([2,100,100,300])
    print(m1.extract_grad())