import torch
from models import lorentz_model
from models import two_p_model
from models import iris_model
import datareader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time

class GPUWorker(object):
    ' Object exists on each GPU and handles individual GPU training '
    def __init__(self,device,pop,batches,top,trunc,mut,maxgen,lw,nw,prob,k,folder):
        ' Constructor downloads parameters and allocates memory for models and data'
        print("LOSS WEIGHT: ",lw, "NOVELTY WEIGHT",nw)
        # User specified parameters
        self.device = torch.device(device)
        self.insertion_probability = prob
        self.novelty_weight = torch.tensor([nw], device=self.device) # Determines how much importance novelty has vs. fitness in ga
        self.loss_weight = torch.tensor([lw], device=self.device)
        self.BC_dim = 5 # Number of BC params equals BC_dim * 3 * num_lorentz oscillators
        self.k = int(pop*k) # kth nearest neighbors in novelty calculation
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

        self.writer = SummaryWriter(folder)

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
        # For 2D parameter scatter animations
        self.mp_gen = [] #Hold model parameters of population for generation
        self.mpe_gen_idx = [] #Holds indices in mp_gen that correspond to elites for generation
        #self.mut_gen_idx = [] #Per generation holds per elite model index in mge_gen_idx parameters of mutated versions
        self.arcv_mod = [] #Holds archived indices in mp_gen that correspond to archived models
        self.arcv_term_idx = []

        print('Memory after basic arrays initialization: ', torch.cuda.memory_allocated(device)/1000000000)

        # BC of population is a list of tensors
        self.BC_pop = torch.empty(pop,1,100,107, device=self.device)  #100,107 ; #2,1 ;  #3, 9
        print('Memory after BC_pop initialization: ', torch.cuda.memory_allocated(device)/1000000000)

        # BC of archive members (arcv_idx tracks end of list as the arcv grows with time)
        # Pre-initialized to be 2% larger than statistically predicted, currently 100% larger
        self.BC_arcv = torch.empty(1,int(pop*self.insertion_probability*maxgen*1.02),100,107, device=self.device) #2,1 ; #100,107
        self.buf = torch.empty(self.num_models,100,107,device=self.device)
        self.arcv_idx = 0

        print('Memory after BC_arcv initialization: ',torch.cuda.memory_allocated(device)/1000000000)

        ' Model generation '
        with torch.no_grad():
            for i in range(pop):
                self.models.append(lorentz_model().cuda(self.device))

            self.furthest_model = [np.zeros_like(self.models[0].p_list())]
            self.avg_tops = []

        print('Memory after model initialization: ', torch.cuda.memory_allocated(device)/1000000000)

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

        print('Memory after data loading: ', torch.cuda.memory_allocated(device)/1000000000)

        'Allocate additional memory if not enough reserved'
        self.rmem = None
        if torch.cuda.memory_reserved(device)/1000000000 <= 9.5:
            print("IF NOT SKIPPED")
            mem = torch.cuda.memory_allocated(device)
            mem_av = 9.5*1000000000
            mem_block = (mem_av - mem)/1000000
            self.rmem = torch.empty((256,1024,int(mem_block)), device=self.device)

        print('Memory after blocking additional memory: ', torch.cuda.memory_allocated(device)/1000000000)
        print('Memory Reserved: ', torch.cuda.memory_reserved(device) / 1000000000)


    def run(self, gen):
        ' Method manages the run on a single gpu '
        with torch.no_grad():
            self.mp_gen.append([])
            self.mpe_gen_idx.append([])
            #self.mut_gen_idx.append([])
            self.avg_tops.append(np.zeros_like(self.models[0].p_list()))

            #for i in range(self.trunc_threshold):
            #    self.mut_gen_idx[gen].append([])


            ' Queue every calculation of fitness for each model on GPU '
            # Generate array of random indices corresponding to each batch for each model
            rand_batch = np.random.randint(self.num_batches,size=self.num_models)
            for j in range(self.num_models):
                g, s = self.train_data[rand_batch[j]]
                fwd, bc = self.models[j](g)
                self.neg_loss[j] = self.fitness_f(fwd,s)
                # calculate and store Behavior Characteristic of each model
                bc = self.models[j] #added for sake of BC=model
                self.mp_gen[gen].append(bc.p_list())
                self.BC_pop[j][0] = lorentz_model.bc_func(bc)

            ' Calculate Novelty Score for each model and add novetly score to fitness score '
            # self.novelty_f loads novelty scores into self.novelty and moves to archive
            self.novelty_f(gen)
            self.fitness = self.neg_loss*self.loss_weight + self.novelty*self.novelty_weight

            ' Insert models into archive with insertion probability '
            # Get bool array of insertion values based on probabilities and random selection of population
            insrt_bools = np.random.random(size=self.num_models) < self.insertion_probability

            # Translate bool array into an array of indices and for loop through these indices
            for i in range(0,self.num_models):
                if insrt_bools[i] == 0:
                    continue
                # Copy and append selected models into archive
                self.Archive.append(self.models[i])
                self.arcv_mod.append(self.models[i].p_list())

                # self.BC_pop stores population info, appended BCs belong to models in the archive
                # Append the BC's of models added to archive to self.BC
                self.BC_arcv[0][self.arcv_idx] = self.BC_pop[i]
                self.arcv_idx += 1
            self.arcv_term_idx.append(self.arcv_idx)

            ' Wait for every kernel queued to execute '
            torch.cuda.synchronize(self.device)

            ' Get sorting based off of fitness array, this function is gpu optimized '
            # Sorted is a tensor of indices organized from best to worst model
            self.sorted = torch.argsort(self.fitness, descending=True)
            self.sorted = torch.argsort(self.fitness, descending=True)

            self.mpe_gen_idx[gen] = self.sorted[0:self.trunc_threshold].cpu().numpy()
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
            rand_model_p = self.models[0].collect_mutations(self.mut,self.num_models,self.device)
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
                #self.mut_gen_idx[gen][rand_top[i-self.trunc_threshold]].append(m.p_list())

            ' Mutate top models that are not champion '
            for i in range(1,self.trunc_threshold):
                # Mutate all elite models except champion
                m = self.models[self.sorted[i]]
                m_cpu = m.p_list()
                self.avg_tops[gen] += m_cpu
                if np.linalg.norm(m_cpu) > np.linalg.norm(self.furthest_model[gen]):
                    self.furthest_model[gen] = m_cpu
                for (mp,mut) in zip(m.parameters(),rand_model_p):
                    # Add random tensor vals drawn from normal distn center=0 var=1, multiplied by mutation power
                    mp.add_(mut[i])
                #self.mut_gen_idx[gen][i].append(m.p_list())

            m = self.models[self.sorted[0]]
            m_cpu = m.p_list()
            self.avg_tops[gen] += m_cpu
            if np.linalg.norm(m_cpu) > np.linalg.norm(self.furthest_model[gen]):
                self.furthest_model[gen] = m_cpu

            self.avg_tops[gen] = self.avg_tops[gen]/self.trunc_threshold
            self.furthest_model.append(self.furthest_model[gen])

            ' Synchronize all operations so that models all mutate and copy before next generation'
            torch.cuda.synchronize(self.device)
            #self.save_plots(gen,plot_arr=[0,9,90])

    def fitness_f(self, x, y, err_ceil=100):
        loss = torch.pow(torch.sub(x, y), 2)
        s0, s1 = loss.size()
        s = s0 * s1
        loss = torch.sum(loss)
        loss = torch.div(loss, s)
        loss[loss != loss] = err_ceil
        return torch.neg(loss)

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
        tot = 10700 #2, #10700 #27

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

            for i in range(self.arcv_idx):
                self.buf = torch.sub(BC_arcv_Table_pop[:,i],BC_arcv_Table_arcv[:,i])
                distances[:,i+self.num_models] = torch.div(torch.sum(torch.pow(self.buf,2),(1,2)),tot)
            self.writer.add_scalar("Archive Variance",
                                   torch.div(torch.sum(torch.narrow(distances,1,self.num_models,self.arcv_idx)), 2*(self.arcv_idx^2)), gen)
            self.writer.add_scalar(
                'Average Novelty in population Models Against those in Archive or Novelty of Archive against population',
                torch.mean(torch.narrow(distances, 1, self.num_models, self.arcv_idx)), gen)

        # Sort all and extra novelty score based on mean of k nearest neighbors
        d_sort, idx = torch.sort(distances)

        # The plus one accounts for the model appearing as its own nearest neighbor once
        neighbors = torch.narrow(d_sort,1,0,self.k+1)
        self.novelty = torch.mean(neighbors,1)

        self.writer.add_scalar('Total Novelty of population Against all Models in Population and Archive', torch.sum(distances), gen)
        self.writer.add_scalar('Total Novelty Among Population in terms of nearest neighbors',torch.div(torch.sum(neighbors),self.k), gen)
        self.writer.add_scalar('Population Variance',
                               torch.div(torch.sum(torch.narrow(distances,1,0,self.num_models)), 2*(self.num_models^2)), gen)