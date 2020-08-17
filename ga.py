import sys, random
import numpy as np
import pickle
import torch
import time
import math
from os.path import join, exists
import multiprocessing
import gc
import copy
import time

class GA:
    def __init__(self, elite_evals, top, threads, timelimit, pop_size, setting, devices, trunc, batches, mut,maxgen):
        '''
        Constructor loads parameters, loads data and models onto each device
        '''
        self.num_batches = batches
        self.top  = top  #Number of top individuals that should be reevaluated
        self.elite_evals = elite_evals  #Number of times should the top individuals be evaluated
        self.model_flags = {'linear' : [8, 100, 100], 'num_spec_point' : 300, 'num_lorentz_osc' : 4,
                            'freq_low' : 0.5,'freq_high' : 5}
        self.pop_size = pop_size
        self.max_gen = maxgen

        ' Block used if GPU Sharing works in OOP manner '
        #' Truncation divisible by two '
        #self.truncation_threshold = int(pop_size*trunc) + 1  #Should be dividable by two

        ' Divide population amongst GPUs and store how many models on each GPU '
        pop_per_device = int( pop_size/len(devices ))
        remainder = int( pop_size%len(devices) )
        self.pop_array = [pop_per_device]*len(devices)
        self.pop_array[1:remainder] += [1]*remainder

        ' Initialize GPUWorkers'
        from train import GPUWorker

        self.devices = devices #Creating single gpu version first
        self.workers = [GPUWorker(dev,self.pop_array[i],batches,top,trunc,elite_evals, mut, self.model_flags,maxgen)
                        for i, dev in enumerate(devices)]


    def run(self, filename, folder):

        ' Genetic Algorithm starts running here '

        'Ideally get rid of while loop and turn it into recursive gpu calls without cpu interference'
        i = 0

        start = time.time()
        while(i<self.max_gen):
            for w in self.workers: w.run(i)
            i += 1
        end = time.time()

        print("Time Elapsed: ", end - start)

        g_file = open(folder + "/fitness_" + filename + ".csv", 'a')
        g_file.close()

        g_file = open(folder + "/fitness_" + filename + ".csv", "r+")
        g_file.truncate(0)
        g_file.close()

        g_file = open(folder + "/fitness_" + filename + ".csv", 'a')
        g_file.write("Generation, elite_fitness\n")

        for w in self.workers:
            arr = w.elite_vals.cpu().numpy()[0]
            for i,val in enumerate(arr):
                g_file.write("{}, {}\n".format(i,val))

        g_file.close()