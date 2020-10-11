import torch

class iris_model(torch.nn.Module):
    def __init__(self, n_in=4, h_nodes_1=50, h_nodes_2=50, n_out=3):
        super(iris_model, self).__init__()
        self.linear = torch.nn.Sequential(torch.nn.Linear(n_in, h_nodes_1),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(h_nodes_1,h_nodes_2),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(h_nodes_2, n_out),
                                     )

    def forward(self, x):
        return self.linear[4](self.linear[3](self.linear[2](self.linear[1](self.linear[0](x)))))

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
            print(mp.size())
            l.append(mp)
        return torch.cat((l[0],l[1].unsqueeze(1),l[2],l[3].unsqueeze(1)), 1)
