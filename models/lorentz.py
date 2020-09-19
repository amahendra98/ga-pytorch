import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P
from torch import pow, add, mul, div
import numpy as np


class network_model(nn.Module):
    def __init__(self,model_flags):
        super(network_model,self).__init__()

        self.linear = model_flags['linear']
        self.num_spec_point = model_flags['num_spec_point']
        self.num_lorentz_osc = model_flags['num_lorentz_osc']
        self.freq_low = model_flags['freq_low']
        self.freq_high = model_flags['freq_high']

        self.linears = nn.ModuleList([])
        self.bn_linears = nn.ModuleList([])
        for ind, fc_num in enumerate(self.linear[0:-1]):  # Excluding the last one as we need intervals
            self.linears.append(nn.Linear(fc_num, self.linear[ind + 1], bias=True))
            self.bn_linears.append(nn.BatchNorm1d(self.linear[ind + 1], track_running_stats=True, affine=True))

        self.lin_w0 = nn.Linear(self.linear[-1], self.num_lorentz_osc, bias=False)
        self.lin_wp = nn.Linear(self.linear[-1], self.num_lorentz_osc, bias=False)
        self.lin_g = nn.Linear(self.linear[-1], self.num_lorentz_osc, bias=False)
        torch.nn.init.uniform_(self.lin_g.weight, a=0.0, b=0.1)

        self.bn_w0 = nn.BatchNorm1d(self.num_lorentz_osc)
        self.bn_wp = nn.BatchNorm1d(self.num_lorentz_osc)
        self.bn_g = nn.BatchNorm1d(self.num_lorentz_osc)

    def forward(self, G):
        out = G
        out = F.relu(self.bn_linears[0](self.linears[0](out))) #GPU Optimized forward maybe?
        out = self.bn_linears[1](self.linears[1](out))

        #out = self.linears[1](F.relu(self.linears[0](out)))

        '''
        for ind, fc in enumerate(self.linears):
            if ind < len(self.linears) - 0:
                out = F.relu(fc(out))
            else:
                out = fc(out)
        
        for ind, (fc, bn) in enumerate(zip(self.linears, self.bn_linears)):
            #print(out.size())
            if ind < len(self.linears) - 0:
                out = F.relu(bn(fc(out)))                                   # ReLU + BN + Linear
            else:
                out = bn(fc(out))
        '''

        #bn version
        w0 = F.relu(self.bn_w0(self.lin_w0(F.relu(out)))).unsqueeze(2) * 1 # size -> [1024,4,1]
        wp = F.relu(self.bn_wp(self.lin_wp(F.relu(out)))).unsqueeze(2) * 1
        g = F.relu(self.bn_g(self.lin_g(F.relu(out)))).unsqueeze(2) * 0.1

        #w0 = F.relu(self.lin_w0(F.relu(out))).unsqueeze(2) * 1 # size -> [1024,4,1]
        #wp = F.relu(self.lin_wp(F.relu(out))).unsqueeze(2) * 1
        #g = F.relu(self.lin_g(F.relu(out))).unsqueeze(2) * 0.1

        # g = torch.sigmoid(self.lin_g(out).unsqueeze(2))

        # Expand them to the make the parallelism, (batch_size, #Lor, #spec_point)
        # self.lin_g.weight.size() = (num_lorent_osc,linear[-1])

        w0 = w0.expand(out.size(0), self.lin_g.weight.size(0), 300)  # size -> [1024,4,300]
        wp = wp.expand_as(w0)
        g = g.expand_as(w0)
        w = torch.arange(0.5, 5, (5 - 0.5) / 300, device=w0.device)
        w_expand = w.expand_as(g)

        e2 = div(mul(pow(wp, 2), mul(w_expand, g)),
        add(pow(add(pow(w0, 2), -pow(w_expand, 2)), 2), mul(pow(w_expand, 2), pow(g, 2))))

        e2 = torch.sum(e2, 1)
        T = e2.float()
        return T #, (w0[0],wp[0],g[0]) #For just one lorentzian

    @staticmethod
    def fitness_f(x, y, err_ceil=100):
        loss = torch.nn.functional.mse_loss(x,y)
        #loss = torch.pow(torch.sub(x, y), 2)
        #s0, s1 = loss.size()
        #s = s0 * s1
        #loss = torch.sum(loss)
        #loss = torch.div(loss, s)
        loss[loss != loss] = err_ceil
        return torch.neg(loss)

    @staticmethod
    def fitness_by_sample(x,y, err_ceil=100):
        loss = torch.pow(torch.sub(x, y), 2)
        loss = torch.div(torch.sum(loss,1), 300)
        return loss
'''
    @staticmethod
    def bc_func(w0, wp, g, steps):
        pass
'''
