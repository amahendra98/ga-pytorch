import torch
from models import lorentz_model
import datareader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class GPUWorker(object):
    ' Object exists on each GPU and handles individual GPU training '
    def __init__(self,device,pop,batches,top,trunc,mut,model_flags,maxgen):
        ' Constructor downloads parameters and allocates memory for models and data'
        self.device = torch.device(device)
        self.num_models = pop
        self.num_batches = batches
        self.models = []
        self.train_data = []
        self.test_data = []
        self.mut = mut
        self.max_gen = maxgen

        # Set trunc threshold to integer
        if top == 0:
            self.trunc_threshold = int(trunc*pop)
        else:
            self.trunc_threshold = top

        self.elite_eval = torch.zeros(self.trunc_threshold, device=self.device)
        self.writer = SummaryWriter("results/P{}_G{}_tr{}".format(pop, maxgen, self.trunc_threshold))

        'Model generation. Theory is created on cpu, moved to gpu, ref stored on cpu'
        for i in range(pop):
            self.models.append(lorentz_model(model_flags).cuda(self.device))

        'Data Storage. Theory is data loaded to cpu, moved to gpu, ref stored on cpu. Would like to store on gpu' \
        'shared memory, however I could not figure out how to from pytorch'
        train_data_loader, test_data_loader = datareader.read_data(x_range=[i for i in range(0, 2)],
                                                                   y_range=[i for i in range(2, 302)],
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
        self.fit = torch.zeros(pop,device=self.device)
        self.sorted = torch.zeros(pop, device=self.device)
        #self.BC = []

    def run(self, gen):
        ' Method manages the run on a single gpu '

        with torch.no_grad():

            'Queue every calculation of fitness for each model on GPU. Doing this all at once with all models already loaded' \
            'Might cause slowdown due to lack of memory. Apparently only 16 or so kernels can execute at once, which I had not' \
            'realized.'
            # Generate array of random indices corresponding to each batch for each model
            rand_batch = np.random.randint(self.num_batches,size=self.num_models)
            for j in range(self.num_models):
                g, s = self.train_data[rand_batch[j]]
                fwd = self.models[j](g)
                self.fit[j] = lorentz_model.fitness_f(fwd,s)
                #self.BC.append(lorentz_model.bc_func(bc))

            ' Wait for every kernel queued to execute '
            torch.cuda.synchronize(self.device)

            ' Get sorting based off of fitness array, this function is gpu optimized '
            # Sorted is a tensor of indices organized from best to worst model
            self.sorted = torch.argsort(self.fit, descending=True)
            self.writer.add_scalar('training loss', self.fit[self.sorted[0]], gen)

            ' Find champion using test data '

            g,s = self.test_data[0]

            # Run all top models through validation set and store fitness
            for i in range(0,self.trunc_threshold):
                self.elite_eval[i] = lorentz_model.fitness_f(self.models[self.sorted[i]](g),s)

            # Sort the evaluation
            self.elite_eval,indices = torch.sort(self.elite_eval, descending=True)

            # Pick champion based on evaluation and fix self.sorted accordingly
            if not(self.sorted[0] == self.elite_eval[0]):
                self.sorted[self.trunc_threshold-1] = self.sorted[0]
                self.sorted[0] = indices[0]

            ' Copy models over truncation barrier randomly into bottom models w/ mutation '
            # Generate array of random indices corresponding to models above trunc barrier, and collect mutation arrays
            rand_model_p = self.collect_random_mutations()
            rand_top = np.random.randint(self.trunc_threshold, size=self.num_models - self.trunc_threshold)
            for i in range(self.trunc_threshold,self.num_models):
                # Grab all truncated models
                m = self.models[self.sorted[i]]

                # Grab random top model
                m_t = self.models[self.sorted[rand_top[i-self.trunc_threshold]]]

                for (mp, mtp, mut) in zip(m.parameters(),m_t.parameters(), rand_model_p):
                    # Copy top model parameters chosen into bottom module and mutate with random tensor of same size
                    # New random tensor vals drawn from normal distn center=0 var=1, multiplied by mutation power
                    mp.copy_(mtp).add_(mut[i])

            ' Mutate top models that are not champion '
            for i in range(1,self.trunc_threshold):
                # Mutate all elite models except champion
                for (mp,mut) in zip(m.parameters(),rand_model_p):
                    # Add random tensor vals drawn from normal distn center=0 var=1, multiplied by mutation power
                    mp.add_(mut[i])

            ' Synchronize all operations so that models all mutate and copy before next generation'
            torch.cuda.synchronize(self.device)
            self.save_plots(gen,plot_arr=[0,9,90])


    def collect_random_mutations(self):
        rand_model_p = []

        rand_lay_1 = torch.mul(torch.randn(self.num_models, 100, 2, requires_grad=False, device=self.device),
                                   self.mut)
        rand_model_p.append(rand_lay_1)

        rand_bi_1 = torch.mul(torch.randn(self.num_models, 100, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bi_1)

        rand_lay_2 = torch.mul(torch.randn(self.num_models, 100, 100, requires_grad=False, device=self.device),
                                   self.mut)
        rand_model_p.append(rand_lay_2)

        rand_bi_2 = torch.mul(torch.randn(self.num_models, 100, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bi_2)

        rand_lin_g = torch.mul(torch.randn(self.num_models, 1, 100, requires_grad=False, device=self.device),
                                   self.mut)
        rand_model_p.append(rand_lin_g)

        rand_lin_w0 = torch.mul(torch.randn(self.num_models, 1, 100, requires_grad=False, device=self.device),
                                    self.mut)
        rand_model_p.append(rand_lin_w0)

        rand_lin_wp = torch.mul(torch.randn(self.num_models, 1, 100, requires_grad=False, device=self.device),
                                    self.mut)
        rand_model_p.append(rand_lin_wp)

        return rand_model_p

    def save_plots(self,gen,rate=20,plot_arr=[0,9,90]):
        if gen % rate == 0:
            fig = plt.figure()

            for subplot,idx in zip(range(1,len(plot_arr)+1,1),plot_arr):
                g, s = self.train_data[0]
                t = g[idx].view((1, 2))
                champ_out = self.models[self.sorted[0]](t)
                w_elite = self.models[self.sorted[self.trunc_threshold - 1]](t)
                worst = self.models[self.sorted[-1]](t)

                ax = fig.add_subplot(1, len(plot_arr), subplot)
                ax.plot(np.linspace(0.5, 5, 300), worst[0].cpu().numpy(), color='tab:red')
                ax.plot(np.linspace(0.5, 5, 300), w_elite[0].cpu().numpy(), color='tab:purple')
                ax.plot(np.linspace(0.5, 5, 300), champ_out[0].cpu().numpy(), color='tab:blue')
                ax.plot(np.linspace(0.5, 5, 300), s[idx].cpu().numpy(), color='tab:orange')

            self.writer.add_figure("{}_champ_worstElite_worst".format(gen), fig)
            plt.close()
