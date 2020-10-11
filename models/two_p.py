import torch

class two_p_model(torch.nn.Module):

    def __init__(self, n_in=1, h_nodes_1=2, n_out=2):
        super(two_p_model, self).__init__()
        self.t1 = torch.nn.Linear(n_in, h_nodes_1, bias=False)
        self.t2 = torch.nn.Sigmoid()
        with torch.no_grad():
            self.t1.weight.data.fill_(0)

    def forward(self, x):
        return self.t2(self.t1(x)), (0,0,0)

    def p_list(self):
        return self.t1.weight.cpu().numpy().transpose()[0]

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
        for mp in z.parameters():
            l = mp
        return l