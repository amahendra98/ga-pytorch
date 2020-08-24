import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P
from torch import pow, add, mul, div
import numpy as np


class lorentz_model(nn.Module):
    def __init__(self,model_flags):
        super(lorentz_model,self).__init__()

        self.linear = model_flags['linear']
        self.num_spec_point = model_flags['num_spec_point']
        self.num_lorentz_osc = model_flags['num_lorentz_osc']
        self.freq_low = model_flags['freq_low']
        self.freq_high = model_flags['freq_high']

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
        out = self.linears[1](F.relu(self.linears[0](out))) #GPU Optimized forward maybe?
        '''
        for ind, fc in enumerate(self.linears):
            if ind < len(self.linears) - 0:
                out = F.relu(fc(out))
            else:
                out = fc(out)
        '''

        w0 = F.relu(self.lin_w0(F.relu(out)))
        wp = F.relu(self.lin_wp(F.relu(out)))
        g = F.relu(self.lin_g(F.relu(out)))

        w0_out = w0
        wp_out = wp
        g_out = g

        w0 = w0.unsqueeze(2) * 1  # size -> [1024,4,1]
        wp = wp.unsqueeze(2) * 1
        g = g.unsqueeze(2) * 0.1
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
        return T, (w0_out,wp_out,g_out) #For just one lorentzian

    @staticmethod
    def fitness_f(x, y, err_ceil=100):
        loss = torch.pow(torch.sub(x, y), 2)
        s0, s1 = loss.size()
        s = s0 * s1
        loss = torch.sum(loss)
        loss = torch.div(loss, s)
        loss[loss != loss] = err_ceil
        return torch.neg(loss)

    @staticmethod
    def bc_func(z, steps, dev): #For single lorentzian only
        (w0, wp, g) = z
        base = int(len(w0)/steps)
        rem = len(w0)%steps

        w0.squeeze()
        wp.squeeze()
        g.squeeze()

        placeholder = 1
        if rem == 0:
            placeholder = 0

        n = torch.zeros(3,placeholder+base*steps,device=dev)

        n[0][0] = torch.mean(w0[0:rem])
        n[1][0] = torch.mean(wp[0:rem])
        n[2][0] = torch.mean(g[0:rem])

        for i in range(base):
            n[0][i+1] = torch.mean(w0[rem + i * steps: rem + (i + 1) * steps])
            n[1][i+1] = torch.mean(w0[rem + i * steps: rem + (i + 1) * steps])
            n[2][i+1] = torch.mean(w0[rem + i * steps: rem + (i + 1) * steps])

        return n

