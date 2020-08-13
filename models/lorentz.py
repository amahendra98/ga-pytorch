import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import pow, add, mul, div
import numpy as np


def fitness_f(x,y):
    loss = nn.functional.mse_loss(x, y, reduction='mean')
    return torch.neg(loss)


class lorentz_model(nn.Module):
    def __init__(self):
        super(lorentz_model,self).__init__()

        self.linear = [8, 100, 100]
        self.num_spec_point = 300
        self.num_lorentz_osc = 4
        self.freq_low = 0.5
        self.freq_high = 5

        w_numpy = np.arange(self.freq_low, self.freq_high, (self.freq_high - self.freq_low) / self.num_spec_point)
        self.w = torch.tensor(w_numpy)
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
        for ind, fc in enumerate(zip(self.linears)):
            if ind < len(self.linears) - 0:
                out = F.relu(fc(out))
            else:
                out = fc(out)

        w0 = F.relu(self.lin_w0(out).unsqueeze(2))  # size -> [1024,4,1]
        wp = F.relu(self.lin_wp(out).unsqueeze(2))
        g = F.relu(self.lin_g(out).unsqueeze(2))
        # g = torch.sigmoid(self.lin_g(out).unsqueeze(2))

        # Expand them to the make the parallelism, (batch_size, #Lor, #spec_point)
        w0 = w0.expand(out.size(0), self.num_lorentz_osc, self.num_spec_point)  # size -> [1024,4,300]
        wp = wp.expand_as(w0)
        g = g.expand_as(w0)
        w_expand = self.w.expand_as(g)

        e2 = div(mul(pow(wp, 2), mul(w_expand, g)),
        add(pow(add(pow(w0, 2), -pow(w_expand, 2)), 2), mul(pow(w_expand, 2), pow(g, 2))))

        e2 = torch.sum(e2, 1)
        T = e2.float()
        return T

