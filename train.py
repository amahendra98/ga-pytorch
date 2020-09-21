import torch
from models import lorentz_model
import datareader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import copy

class GPUWorker(object):
    ' Object exists on each GPU and handles individual GPU training '
    def __init__(self,device,pop,batches,top,trunc,mut,model_flags,maxgen,lw,nw,prob):
        ' Constructor downloads parameters and allocates memory for models and data'
        print("LOSS WEIGHT: ",lw, "NOVELTY WEIGHT",nw)
        # User specified parameters
        self.device = torch.device(device)
        self.insertion_probability = prob
        self.novelty_weight = torch.tensor([nw], device=self.device) # Determines how much importance novelty has vs. fitness in ga
        self.loss_weight = torch.tensor([lw], device=self.device)
        self.BC_dim = 5 # Number of BC params equals BC_dim * 3 * num_lorentz oscillators
        self.k = 25 # kth nearest neighbors in novelty calculation
        self.num_models = pop
        self.num_batches = batches
        self.mut = mut
        self.max_gen = maxgen
        if top == 0 and trunc != 0:
            self.trunc_threshold = int(trunc*pop)
        elif trunc == 0 and top !=0:
            self.trunc_threshold = top
        else:
            raise Exception("Either top or trunc must equal 0, but never both")

        (y,m,d,hr,min,s,x1,x2,x3) = time.localtime(time.time())
        self.writer = SummaryWriter("results/top{}_insrt{}_lw{}_nw{}_{}{}{}_{}{}{}".format(top,prob,lw,nw,y,m,d,hr,min,s))

        # Parameters to hold memory
        self.models = []
        self.train_data = []
        self.test_data = []
        self.elite_eval = torch.zeros(self.trunc_threshold, device=self.device)
        self.fitness = torch.zeros(pop,device=self.device)
        self.neg_loss = torch.zeros(pop, device=self.device)
        self.sorted = torch.zeros(pop, device=self.device)
        self.novelty = torch.zeros(pop, device=self.device)
        self.Archive = []
        print('Memory after basic arrays initialization: ', torch.cuda.memory_allocated('cuda:1')/1000000000)

        # BC of population is a list of tensors
        self.BC_pop = torch.empty(pop,1,100,107, device=self.device)
        print('Memory after BC_pop initialization: ', torch.cuda.memory_allocated('cuda:1')/1000000000)

        # BC of archive members (arcv_idx tracks end of list as the arcv grows with time)
        # Pre-initialized to be 5% larger than statistically predicted
        self.BC_arcv = torch.empty(1,int(pop*self.insertion_probability*maxgen*1.05),100,107, device=self.device)
        self.buf = torch.empty_like(self.BC_arcv[0])
        self.arcv_idx = 0

        print('Memory after BC_arcv initialization: ',torch.cuda.memory_allocated('cuda:1')/1000000000)

        ' Model generation '
        for i in range(pop):
            self.models.append(lorentz_model(model_flags).cuda(self.device))

        print('Memory after model initialization: ', torch.cuda.memory_allocated('cuda:1')/1000000000)

        ' Data Loading '
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

        print('Memory after data loading: ', torch.cuda.memory_allocated('cuda:1')/1000000000)

    def run(self, gen):
        ' Method manages the run on a single gpu '

        with torch.no_grad():
            ' Queue every calculation of fitness for each model on GPU '
            # Generate array of random indices corresponding to each batch for each model
            rand_batch = np.random.randint(self.num_batches,size=self.num_models)
            for j in range(self.num_models):
                g, s = self.train_data[rand_batch[j]]
                fwd, bc = self.models[j](g)
                self.neg_loss[j] = lorentz_model.fitness_f(fwd,s)
                # calculate and store Behavior Characteristic of each model
                bc = self.models[j] #added for sake of BC=model
                self.BC_pop[j][0] = lorentz_model.bc_func(bc,self.BC_dim,self.device)

            ' Calculate Novelty Score for each model and add novetly score to fitness score '
            # self.novelty_f loads novelty scores into self.novelty and moves to archive
            self.novelty_f(gen)
            self.fitness = self.neg_loss*self.loss_weight + self.novelty*self.novelty_weight

            ' Insert models into archive with insertion probability '
            # Get bool array of insertion values based on probabilities and random selection of population
            insrt_bools = np.random.random(size=self.num_models) < self.insertion_probability

            # Translate bool array into an array of indices and for loop through these indices
            for i in np.linspace(0,self.num_models-1,self.num_models,dtype=np.int32)[insrt_bools]:
                # Copy and append selected models into archive
                self.Archive.append(self.models[i])

                # self.BC_pop stores population info, appended BCs belong to models in the archive
                # Append the BC's of models added to archive to self.BC
                self.BC_arcv[0][self.arcv_idx] = self.BC_pop[i]
                self.arcv_idx += 1

            ' Wait for every kernel queued to execute '
            torch.cuda.synchronize(self.device)

            ' Get sorting based off of fitness array, this function is gpu optimized '
            # Sorted is a tensor of indices organized from best to worst model
            self.sorted = torch.argsort(self.fitness, descending=True)
            self.writer.add_scalar('training values of pure negative mseloss over generation', self.neg_loss[self.sorted[0]],gen)
            self.writer.add_scalar('training values of 1000 * novelty score over generation', 1000*self.novelty[self.sorted[0]],gen)
            self.writer.add_scalar('training values of fitness function over generation', self.fitness[self.sorted[0]], gen)

            ' Find champion by validating against test data set (compromises test data set for future eval)'
            '''
            g,s = self.test_data[0]

            # Run top training model over evaluation dataset to get elite_val fitness score and index
            fwd, bc = self.models[self.sorted[0]](g)
            elite_val = lorentz_model.fitness_f(fwd,s)
            self.elite_eval[0] = elite_val

            # Run rest of top models over evaluation dataset, storing their scores and saving champion and its index
            for i in range(1,self.trunc_threshold):
                fwd,bc = self.models[self.sorted[i]](g)
                self.elite_eval[i] = lorentz_model.fitness_f(fwd, s)
                if elite_val < self.elite_eval[i]:
                    # Swap current champion index in self.sorted with index of model that out-performed it in eval
                    # Technically the sorted list is no longer in order, but this does not matter as the top models
                    # are all still top models and the champion is in the champion position
                    elite_val = self.elite_eval
                    former_champ_idx = self.sorted[0]
                    self.sorted[0] = self.sorted[i]
                    self.sorted[i] = former_champ_idx
            '''

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
        ' Pre-loads mutation arrays so that rand is not called in for loop multiple times '

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
        ' Plots champion, worst elite, and worst graph against each graph indexed from the raw training data '
        if gen % rate == 0:
            fig = plt.figure()

            for subplot,idx in zip(range(1,len(plot_arr)+1,1),plot_arr):
                g, s = self.train_data[0]
                t = g[idx].view((1, 2))
                champ_out = self.models[self.sorted[0]](t)
                w_elite = self.models[self.sorted[self.trunc_threshold - 1]](t)
                worst = self.models[self.sorted[-1]](t)

                ax = fig.add_subplot(1, len(plot_arr), subplot)
                ax.plot(np.linspace(0.5, 5, 300), worst[0].cpu().numpy()[0], color='tab:red')
                ax.plot(np.linspace(0.5, 5, 300), w_elite[0].cpu().numpy()[0], color='tab:purple')
                ax.plot(np.linspace(0.5, 5, 300), champ_out[0].cpu().numpy()[0], color='tab:blue')
                ax.plot(np.linspace(0.5, 5, 300), s[idx].cpu().numpy(), color='tab:orange')

            self.writer.add_figure("{}_champ_worstElite_worst".format(gen), fig)


    def novelty_f(self, gen):
        ' Calculates novelty score of all models in one function '
        # Elements in model
        tot = 101*107

        distances = torch.empty((self.num_models,self.num_models+self.arcv_idx), device=self.device)

        # Given list of each model's BC, expand list into a square matrix. Then get its transpose
        BC_pop_Table = self.BC_pop.expand(-1,self.num_models,-1,-1)
        tran = BC_pop_Table.transpose(0,1)

        # mse_loss of these two matrices will give mse_loss between each model against every other model
        # For loop handles job iteratively so gpu does not run out of memory
        for i in range(self.num_models):
            distances[i][0:self.num_models] = torch.div(torch.sum(torch.pow(torch.sub(BC_pop_Table[i], tran[i]),2),(1,2)),tot)

        # If archive holds models, repeat steps above, except with BC's of models in archive
        if self.arcv_idx > 0:
            ret = 0
            if self.arcv_idx == 0:
                ret = 1
            self.arcv_idx += ret

            BC_arcv_Table_pop = self.BC_pop.expand(-1,self.arcv_idx,-1,-1)
            BC_arcv_Table_arcv = torch.narrow(self.BC_arcv.expand(self.num_models,-1,-1,-1),1,0,self.arcv_idx)

            self.arcv_idx -= ret

            for i in range(self.num_models):
                self.buf = torch.sub(BC_arcv_Table_pop[i], BC_arcv_Table_arcv[i])
                distances[i][self.num_models:self.num_models+self.arcv_idx] = torch.div(torch.sum(torch.pow(self.buf,2),(1,2)),tot)

            self.writer.add_scalar(
                'Total Novelty in population Models Against those in Archive or Novelty of Archive against population',
                torch.sum(torch.narrow(distances, 1, self.num_models, self.arcv_idx)), gen)

        # Sort all and extra novelty score based on mean of k nearest neighbors
        d_sort, idx = torch.sort(distances)

        # The plus one accounts for the model appearing as its own nearest neighbor once
        neighbors = torch.narrow(d_sort,1,0,self.k+1)
        self.novelty = torch.mean(neighbors,1)

        self.writer.add_scalar('Total Novelty of population Against all Models in Population and Archive', torch.sum(distances), gen)
        self.writer.add_scalar('Total Novelty Among Population in terms of nearest neighbors',torch.sum(neighbors), gen)
        self.writer.add_scalar('Total Novelty of population Against all Other models in Population',
                               torch.sum(torch.narrow(distances,1,0,self.num_models)), gen)