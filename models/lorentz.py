import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter as P
from torch import pow, add, mul, div
import numpy as np


class lorentz_model(nn.Module):
    def __init__(self, linear):
        super(lorentz_model,self).__init__()
        self.linear = linear[0:-1]
        self.num_spec_point = linear[-1]
        self.num_lorentz_osc = 1
        self.freq_low = 0.5
        self.freq_high = 5

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
        w = torch.arange(0.5, 5, (5 - 0.5) / 300, device=w0.device)
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
                RAND.append(torch.mul(torch.randn(num,s[0],requires_grad=False,device=dev), mut))
            if len(s) == 2:
                RAND.append(torch.mul(torch.randn(num,s[0],s[1],requires_grad=False, device=dev), mut))
        return RAND

    @staticmethod
    def bc_func(z): #For single lorentzian only
        l = []
        for mp in z.parameters():
            l.append(mp)
        return torch.cat((l[0],l[1].unsqueeze(1),l[2],l[3].unsqueeze(1),l[4][0].unsqueeze(1),l[5][0].unsqueeze(1),
                          l[6][0].unsqueeze(1)), 1)

    def p_list(self):
        return lorentz_model.bc_func(self).cpu().numpy()