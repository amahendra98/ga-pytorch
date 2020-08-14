import math 
import torch
from torchvision import transforms
from multiprocessing import Lock
import numpy as np
from models import lorentz_model, fitness_f
import gym
import copy
import datareader
import random

class GPUWorker(object):
    ' Object exists on each GPU and handles individual GPU training '
    def __init__(self,device,pop,batches,top,trunc,elite_evals,mut,model_flags,maxgen):
        ' Constructor downloads parameters and allocates memory for models and data'
        self.device = torch.device(device)
        self.top = top
        self.elite_evals = elite_evals
        self.num_models = pop
        self.num_batches = batches
        self.models = []
        self.train_data = []
        self.test_data = []
        self.mut = mut
        self.max_gen = maxgen

        # Set trunc threshold to even integer
        self.trunc_threshold = int(trunc*pop) + 1
        if self.trunc_threshold%2 == 1:
            self.trunc_threshold - 1

        'Model generation. Theory is created on cpu, moved to gpu, ref stored on cpu'
        for i in range(pop):
            self.models.append(lorentz_model(model_flags).cuda(self.device))

        'Data Storage. Theory is data loaded to cpu, moved to gpu, ref stored on cpu. Would like to store on gpu' \
        'shared memory, however I could not figure out how to from pytorch'
        train_data_loader, test_data_loader = datareader.read_data(x_range=[i for i in range(0, 8)],
                                                                   y_range=[i for i in range(8, 308)],
                                                                   geoboundary=[20, 200, 20, 100],
                                                                   batch_size=0,
                                                                   set_size=batches,
                                                                   normalize_input=True,
                                                                   data_dir='./',
                                                                   test_ratio=0.2)

        for i, (geometry, spectra) in enumerate(train_data_loader):
            self.train_data.append( (geometry.to(self.device),spectra.to(self.device)) )

        for i, (geometry, spectra) in enumerate(test_data_loader):
            self.test_data.append( (geometry.to(self.device),spectra.to(self.device)) )

        'GPU Tensor that stores fitness values & sorts from population. Would Ideally store in gpu shared memory'
        self.fit = torch.zeros(1,pop,device=self.device)
        self.sorted = torch.zeros(1, pop, device=self.device)

        'Create tensor stored on gpu to store eilte values over generations'
        self.elite_vals = torch.zeros(1,maxgen, device=self.device)

    def run(self, gen):
        ' Method manages the run on a single gpu '

        with torch.no_grad():

            'Queue every calculation of fitness for each model on GPU. Doing this all at once with all models already loaded' \
            'Might cause slowdown due to lack of memory. Apparently only 16 or so kernels can execute at once, which I had not' \
            'realized.'
            for i,m in enumerate(self.models): #Might want a gpu version of counting to avoid enumerate.
                g,s = self.train_data[torch.randint(0,self.num_batches,(1,1),device=self.device)[0][0]]
                self.fit[0][i] = fitness_f(m(g),s)

            ' Wait for every kernel queued to execute '
            torch.cuda.synchronize(self.device)

            ' Get sorting based off of fitness array, this function is gpu optimized '
            # Sorted is a tensor of indices organized from best to worst model
            self.sorted = torch.argsort(self.fit, descending=True)
            self.elite_vals[0][gen] = self.fit[0][self.sorted[0][0]]

            ' Find elite and do evaluation '
            # Step skipped for now, should involve test set evaluation

            ' Copy top models randomly into bottom models w/ mutation '
            for idx in self.sorted[0][self.top:-1]:
                'Theory: CPU involved in for loop and calls to self only, while self not stored on gpu'
                # Grab a model from an index between top and end
                m = self.models[idx]

                # Get random index i between 0 and top
                # Put index into sorted[i] to get index of a top model
                # Copy top model into place of a bottom model
                m_t = self.models[self.sorted[0][torch.randint(0,self.top,(1,1),device=self.device)[0][0]]]

                # Copy top model parameters chosen into bottom module
                m.linears[0].weight.copy_(m_t.linears[0].weight)
                m.linears[1].weight.copy_(m_t.linears[1].weight)
                m.linears[0].bias.copy_(m_t.linears[0].bias)
                m.linears[1].bias.copy_(m_t.linears[1].bias)
                m.lin_g.weight.copy_(m_t.lin_g.weight)
                m.lin_wp.weight.copy_(m_t.lin_wp.weight)
                m.lin_w0.weight.copy_(m_t.lin_w0.weight)

                # Access underlying model tensor (might not involve additional cpu calls)
                # Create a new random tensor like it with val drawn from normal distn center=0 var=1
                # Add random tensor to underlying model tensor in place
                m.linears[0].weight.add_(torch.mul(torch.randn_like(m.linears[0].weight,device=self.device),self.mut))
                m.linears[0].bias.add_(torch.mul(torch.randn_like(m.linears[0].bias, device=self.device),self.mut))
                m.linears[1].weight.add_(torch.mul(torch.randn_like(m.linears[1].weight, device=self.device),self.mut))
                m.linears[1].bias.add_(torch.mul(torch.randn_like(m.linears[1].bias, device=self.device),self.mut))

            ' Mutate top models except for elite '
            for idx in self.sorted[0][1:self.top]:
                m = self.models[idx]
                m.linears[0].weight.add_(torch.mul(torch.randn_like(m.linears[0].weight,device=self.device), self.mut))
                m.linears[0].bias.add_(torch.mul(torch.randn_like(m.linears[0].bias, device=self.device),self.mut))
                m.linears[1].weight.add_(torch.mul(torch.randn_like(m.linears[1].weight, device=self.device),self.mut))
                m.linears[1].bias.add_(torch.mul(torch.randn_like(m.linears[1].bias, device=self.device),self.mut))
                m.lin_g.weight.add_(torch.mul(torch.randn_like(m.lin_g.weight, device=self.device),self.mut))
                m.lin_wp.weight.add_(torch.mul(torch.randn_like(m.lin_wp.weight, device=self.device), self.mut))
                m.lin_w0.weight.add_(torch.mul(torch.randn_like(m.lin_w0.weight, device=self.device), self.mut))

            ' Synchronize all operations so that models all mutate and copy before next generation'
            torch.cuda.synchronize(self.device)