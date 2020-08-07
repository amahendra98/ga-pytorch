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
    def __init__(self, device, num_models):
        self.train_data = []
        self.test_data = []
        self.device = device
        self.num_models = num_models
        self.id_list = [0] * num_models

        # Create one batch for each model being trained using this "DataStore"
        train_data_loader, test_data_loader = datareader.read_data(x_range=[i for i in range(0, 8)],
                                                                   y_range=[i for i in range(8, 308)],
                                                                   geoboundary=[20, 200, 20, 100],
                                                                   batch_size=0,
                                                                   set_size=1,#self.num_models,
                                                                   normalize_input=True,
                                                                   data_dir='./',
                                                                   test_ratio=0.2)

        # Get DataSets into arrays of tuples stored once on desired device (avoids memory intensive pytorch iterators)
        for i, (geometry, spectra) in enumerate(train_data_loader):
            self.train_data.append( (geometry.to(self.device),spectra.to(self.device)) )

        for i, (geometry, spectra) in enumerate(test_data_loader):
            self.test_data.append( (geometry.to(self.device),spectra.to(self.device)) )

    def get_batch(self, ident):
        ident = int(ident)
        idx = ident + self.id_list[ident]
        #if self.num_models < self.id_list[ident]:
        #    print("Done!")
        #    return None
        idx = idx%self.num_models
        self.id_list[ident] += 1
        print('got data with index: ', idx, '\tfrom id: ',ident)
        return self.train_data[idx],self.test_data[idx]


"""
    def train_iter(self):
        self.train_loader_iter = iter(self.train_loader)

    def test_iter(self):
        self.test_loader_iter = iter(self.test_loader)

    def get_train_batch(self):
        batch_info = next(self.train_loader_iter)
        self.train_iter()
        return batch_info

    def get_test_batch(self):
        batch_info = next(self.test_loader_iter)
        self.test_iter()
        return batch_info
"""


class RolloutGenerator(object):
    """ Utility to generate rollouts.
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, store, ident):
        self.lorentz = lorentz_model()
        self.dataSource = store
        self.id = ident

    def do_rollout(self):
        with torch.no_grad():
            result = self.dataSource.get_batch(self.id)
            test_fitness = 0
            train_fitness = 0
            for i in range(self.dataSource.num_models):
                (train_batch, test_batch) = result
                (geometry_train, spectra_train) = train_batch
                (geometry_test, spectra_test) = test_batch

                lorentz_train_graph = self.lorentz(geometry_train)
                lorentz_test_graph = self.lorentz(geometry_test)

                train_fitness += fitness_f(lorentz_train_graph, spectra_train)
                test_fitness += fitness_f(lorentz_test_graph, spectra_test)
            self.dataSource.id_list[int(self.id)] = 0
            return train_fitness,test_fitness

def fitness_eval_parallel(pool, r_gen):#, controller_parameters):
    return pool.apply_async(r_gen.do_rollout)


class GAIndividual():
    '''
    GA Individual

    multi = flag to switch multiprocessing on or off
    '''
    def __init__(self, ident, device, time_limit, setting, store, multi=True):
        self.id = ident
        self.device = device
        self.time_limit = time_limit #Unnecessary
        self.multi = multi
        self.mutation_power = 0.01
        self.setting = setting
        self.r_gen = RolloutGenerator(store, ident)
        #self.r_gen.discrete_VAE = self.discrete_VAE

        self.async_results = []
        self.calculated_results = {}

    def run_solution(self, pool,evals=5, force_eval=False):
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
        if (evals in self.calculated_results.keys()): #Already calculated?
            mean_training_fitness, mean_testing_fitness = self.calculated_results[evals]
        else:
            if self.multi:
                results = [t.get() for t in self.async_results]
            else:
                results = [t for t in self.async_results]

            mean_training_fitness = np.mean ( [train_f[0] for train_f in results] )
            mean_testing_fitness = np.mean( [test_f[1] for test_f in results] )
            #std_training_fitness = np.std( results )

            self.calculated_results[evals] = (mean_training_fitness, mean_testing_fitness)

        self.fitness = -mean_training_fitness
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