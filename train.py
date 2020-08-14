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

class DataStore(object):
    def __init__(self, device, num_models,num_batches):
        """ Class manages data. Objects of this class are loaded once per device in use. Rollout objects query
        data from source as necessary """

        self.train_data = []
        self.test_data = []
        self.device = device
        self.num_models = num_models
        self.num_batches = num_batches

        # Create one batch for each model being trained using this "DataStore"
        train_data_loader, test_data_loader = datareader.read_data(x_range=[i for i in range(0, 8)],
                                                                   y_range=[i for i in range(8, 308)],
                                                                   geoboundary=[20, 200, 20, 100],
                                                                   batch_size=0,
                                                                   set_size=self.num_batches,
                                                                   normalize_input=True,
                                                                   data_dir='./',
                                                                   test_ratio=0.2)

        # Get DataSets into arrays of tuples stored once on a desired device (avoids memory intensive pytorch iterators)
        for i, (geometry, spectra) in enumerate(train_data_loader):
            self.train_data.append( (geometry.to(self.device),spectra.to(self.device)) )

        for i, (geometry, spectra) in enumerate(test_data_loader):
            self.test_data.append( (geometry.to(self.device),spectra.to(self.device)) )

    def get_batch(self, idx):
        return self.train_data[idx],self.test_data[idx]

class RolloutGenerator(object):
    """ Class that connects to dataSource and grabs data, as well as holding and training models """
    def __init__(self, store):
        self.lorentz = lorentz_model()
        self.dataSource = store

    def do_rollout(self):
        ' Function trains models and gets fitness end to end '

        with torch.no_grad():
            ' Batches are randomly selected '
            result = self.dataSource.get_batch(random.choice(range(self.dataSource.num_batches)))

            (train_batch, test_batch) = result
            (geometry_train, spectra_train) = train_batch
            (geometry_test, spectra_test) = test_batch

            lorentz_train_graph = self.lorentz(geometry_train)
            lorentz_test_graph = self.lorentz(geometry_test)

            train_fitness = fitness_f(lorentz_train_graph, spectra_train)
            test_fitness = fitness_f(lorentz_test_graph, spectra_test)

            return train_fitness,test_fitness

def fitness_eval_parallel(pool, r_gen):#, controller_parameters):
    return pool.apply_async(r_gen.do_rollout)


class GAIndividual():
    '''
    GA Individual

    multi = flag to switch multiprocessing on or off
    '''
    def __init__(self, device, time_limit, setting, store, multi=True):
        self.device = device
        self.time_limit = time_limit #Unnecessary
        self.multi = multi
        self.mutation_power = 0.02
        self.setting = setting
        self.r_gen = RolloutGenerator(store)
        #self.r_gen.discrete_VAE = self.discrete_VAE

        self.async_results = []
        self.calculated_results = {}

    def run_solution(self, pool,evals=5, force_eval=False):
        'Checks if desired output already calculated, and if not sends task to pool'
        if force_eval:
            self.calculated_results.pop(evals, None)
        if (evals in self.calculated_results.keys()): #Already caculated results
            return

        self.async_results = []

        for i in range(evals):
            if self.multi:
                self.async_results.append (fitness_eval_parallel(pool, self.r_gen))#, self.controller_parameters) )
            else:
                self.async_results.append (self.r_gen.do_rollout() )


    def evaluate_solution(self, evals):
        ' Grabs task from pool and returns fitness data '

        print('Task called')
        if (evals in self.calculated_results.keys()): #Already calculated?
            mean_training_fitness, mean_testing_fitness = self.calculated_results[evals]
        else:
            if self.multi:
                results = [t.get() for t in self.async_results]
            else:
                results = [t for t in self.async_results]

            print('eval_solution: ',results)
            mean_training_fitness = np.mean ( [train_f[0] for train_f in results] )
            mean_testing_fitness = np.mean( [test_f[1] for test_f in results] )
            #std_training_fitness = np.std( results )

            self.calculated_results[evals] = (mean_training_fitness, mean_testing_fitness)

        self.fitness = -mean_training_fitness
        self.test_fitness = -mean_testing_fitness
        return mean_training_fitness, mean_testing_fitness


    def load_solution(self, filename):
        s = torch.load(filename)
        self.r_gen.lorentz.load_state_dict( s['lorentz'] )

    def clone_individual(self,params):
        child_solution = GAIndividual(params[0], params[1], self.time_limit, self.setting, params[2], multi=True)
        child_solution.multi = self.multi
        child_solution.fitness = self.fitness
        child_solution.r_gen.lorentz = copy.deepcopy (self.r_gen.lorentz)
        return child_solution
    
    def mutate_params(self, params):
        for key in params:
            if key in ['bn_linears.{}.num_batches_tracked'.format(i) for i in (0,1)]:
                continue
            tmp = np.random.normal(0, 1, params[key].size()) * self.mutation_power
            params[key] += torch.from_numpy(tmp).float()

    def mutate(self):
        self.mutate_params(self.r_gen.lorentz.state_dict())