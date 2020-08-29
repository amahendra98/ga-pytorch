import os
os.environ['PATH'] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

import sys
sys.path.append('C:/Program Files/Graphviz 2.44.1/bin/dot.exe')

import time

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
        (y,m,d,hr,min,s,x1,x2,x3) = time.localtime(time.time())
        self.writer = SummaryWriter("results/P{}_G{}_tr{}_{}{}{}_{}{}{}".format(pop, maxgen, self.trunc_threshold,y,m,d,
                                                                                hr,min,s))

        'Model generation. Created on cpu, moved to gpu, ref stored on cpu'
        for i in range(pop):
            self.models.append(lorentz_model(model_flags).cuda(self.device))

        'Data set Storage'
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

        # Load in best_model.pt and start a population of its mutants
        with torch.no_grad():
            rand_mut = self.collect_random_mutations()
            self.models[0] = torch.load('best_model.pt', map_location=self.device)

            m_t = self.models[0]
            for i in range(0, pop):
                m = self.models[i]
                for (mp, m_tp, mut) in zip(m.parameters(), m_t.parameters(), rand_mut):
                    mp.copy_(m_tp).add_(mut[i])


        'GPU Tensor that stores fitness values & sorts from population. Would Ideally store in gpu shared memory'
        self.fit = torch.zeros(pop,device=self.device)
        self.sorted = torch.zeros(pop, device=self.device)
        self.hist_plot = []
        self.lorentz_plot = []


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
                self.models[j].eval()
                fwd = self.models[j](g)
                self.fit[j] = lorentz_model.fitness_f(fwd,s)
                #self.BC.append(lorentz_model.bc_func(bc))

            ' Wait for every kernel queued to execute '
            torch.cuda.synchronize(self.device)

            ' Get sorting based off of fitness array, this function is gpu optimized '
            # Sorted is a tensor of indices organized from best to worst model
            self.sorted = torch.argsort(self.fit, descending=True)
            self.writer.add_scalar('training loss', self.fit[self.sorted[0]], gen)

            ' Find champion by validating against test data set (compromises test data set for future eval)'
            g,s = self.test_data[0]

            # Run top training model over evaluation dataset to get elite_val fitness score and index
            elite_val = lorentz_model.fitness_f(self.models[self.sorted[0]](g),s)
            self.elite_eval[0] = elite_val

            # Run rest of top models over evaluation dataset, storing their scores and saving champion and its index
            for i in range(1,self.trunc_threshold):
                self.elite_eval[i] = lorentz_model.fitness_f(self.models[self.sorted[i]](g), s)
                if elite_val < self.elite_eval[i]:
                    # Swap current champion index in self.sorted with index of model that out-performed it in eval
                    # Technically the sorted list is no longer in order, but this does not matter as the top models
                    # are all still top models and the champion is in the champion position
                    elite_val = self.elite_eval[i]
                    former_champ_idx = self.sorted[0]
                    self.sorted[0] = self.sorted[i]
                    self.sorted[i] = former_champ_idx

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
            self.save_plots(gen,plot_arr=[0,18,36,63])


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

        #Batchnorm mods
        rand_bn_lin_1 =torch.mul(torch.randn(self.num_models, 100, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_lin_1)

        rand_bn_bias_1 =torch.mul(torch.randn(self.num_models, 100, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_bias_1)

        rand_bn_lin_2 =torch.mul(torch.randn(self.num_models, 100, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_lin_2)

        rand_bn_bias_2 =torch.mul(torch.randn(self.num_models, 100, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_bias_2)

        rand_lin_g = torch.mul(torch.randn(self.num_models, 1, 100, requires_grad=False, device=self.device),
                                   self.mut)
        rand_model_p.append(rand_lin_g)

        rand_lin_w0 = torch.mul(torch.randn(self.num_models, 1, 100, requires_grad=False, device=self.device),
                                    self.mut)
        rand_model_p.append(rand_lin_w0)

        rand_lin_wp = torch.mul(torch.randn(self.num_models, 1, 100, requires_grad=False, device=self.device),
                                    self.mut)
        rand_model_p.append(rand_lin_wp)

        #Batchnorm mods
        rand_bn_w0_w =torch.mul(torch.randn(self.num_models, 1, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_w0_w)
        rand_bn_w0_b =torch.mul(torch.randn(self.num_models, 1, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_w0_b)
        rand_bn_wp_w =torch.mul(torch.randn(self.num_models, 1, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_wp_w)
        rand_bn_wp_b =torch.mul(torch.randn(self.num_models, 1, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_wp_b)
        rand_bn_g_w =torch.mul(torch.randn(self.num_models, 1, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_g_w)
        rand_bn_g_b =torch.mul(torch.randn(self.num_models, 1, requires_grad=False, device=self.device),
                                  self.mut)
        rand_model_p.append(rand_bn_g_b)

        return rand_model_p

    def save_plots(self,gen,rate=10,plot_arr=[0,9,18,27,36,45,54,63,72,81,90]):
        if gen % rate == 0:
            fig = plt.figure()

            g, s = self.train_data[0]
            for subplot,idx in zip(range(1,len(plot_arr)+1,1),plot_arr):
                t = g[idx].view((1, 2))
                champ = self.models[self.sorted[0]]
                champ_out = champ(t)[0].cpu().numpy()
                w_elite = self.models[self.sorted[self.trunc_threshold - 1]]
                w_elite_out = w_elite(t)[0].cpu().numpy()
                worst = self.models[self.sorted[-1]]
                worst_out = worst(t)[0].cpu().numpy()
                sn = s[idx].cpu().numpy()

                self.lorentz_plot.append(champ_out)
                self.lorentz_plot.append(w_elite_out)
                self.lorentz_plot.append(worst_out)
                self.lorentz_plot.append(sn)

                ax = fig.add_subplot(1, len(plot_arr), subplot)
                ax.plot(np.linspace(0.5, 5, 300), worst_out, color='tab:red', label='worst')
                ax.plot(np.linspace(0.5, 5, 300), w_elite_out, color='tab:purple',label='worst elite')
                ax.plot(np.linspace(0.5, 5, 300), champ_out, color='tab:blue', label='Champion')
                ax.plot(np.linspace(0.5, 5, 300), sn, color='tab:orange',label='Truth Spectra')
            plt.legend()
            plt.xlabel("Frequency (THz)")
            plt.ylabel("e2")

            self.writer.add_figure("{}_Lorentz_Evolution".format(gen), fig)

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            champ_out = self.models[self.sorted[0]](g)
            hist_ydata = lorentz_model.fitness_by_sample(champ_out, s).cpu().numpy()
            self.hist_plot.append(hist_ydata)
            ax.hist(hist_ydata, bins=np.arange(0,5,0.05))
            #ax.set_xticks([0,0.1,0.3,0.5,0.8,1,2,5,10,20,50])
            #plt.xscale("log")
            plt.xlabel("MSE_Loss over training")
            plt.ylabel("Number of datasets")

            self.writer.add_figure("{}_Histogram".format(gen), fig)
            plt.close()