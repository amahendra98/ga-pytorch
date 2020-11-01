import flag_reader
from schedulers import *
import torch
import numpy as np
import time

class GA:
    def __init__(self, flags):
        self.flags = flags
        self.max_gen = flags.generations
        devices = flags.device[0]
        pop_size = flags.pop_size
        self.name = flags.folder
        self.gen = 0
        self.sched = None

        func_args = flags.schedule_args[1:]
        if flags.schedule_args[0] == 'generational_scheduler':
            self.sched = generational_scheduler(*func_args)
        if flags.schedule_args[0] == 'value_based_scheduler':
            self.sched = value_based_scheduler(*func_args)
        if flags.schedule_args[0] == 'step_length_doubling_scheduler':
            self.sched = step_length_doubling_scheduler(*func_args)
        if flags.schedule_args[0] == 'variable_length_value_scheduler':
            self.sched = variable_length_value_scheduler(*func_args)

        ' Divide population amongst GPUs and store how many models on each GPU '
        pop_per_device = int( pop_size/len(devices ))
        remainder = int( pop_size%len(devices) )
        self.pop_array = [pop_per_device]*len(devices)
        self.pop_array[1:remainder] += [1]*remainder

        ' Initialize GPUWorkers'
        from train import GPUWorker
        self.devices = devices #Creating single gpu version first
        self.worker = GPUWorker(flags)

    def run(self):
        ' DO GA ALGORITHM '

        ' Clear the space occupied by rmem (space was occupied to force pytorch to allocate enough memory for run'
        self.worker.rmem = None

        ' Time and execute run over max generations'
        start = time.time()
        print(self.name)

        for gen in range(self.max_gen):
            self.step()

        '''
        lr_sched = [(250,1)]#[(0,0.05), (20,0.02), (40, 0.01), (100,0.005), (200, 0.001), (300, 0.0005)]
        for gz in range(self.max_gen):
            for step, lr in lr_sched:
                if gz > step and self.worker.mut > lr:
                    self.worker.mut = lr
                    print("At Gen: ", gz, " mutation power changed to ", self.worker.mut)
            self.worker.run(gz)
            print("Gen: ", gz, "Mutation Rate: ", self.worker.mut)
        '''

        ' Rate run '
        elapsed = time.time() - start

        # Finds distance between origin model of run and center mean of all models after run
        #metric, avg_top, furthest = self.metric_center()
        metric = self.best_validation_loss()


        ' Store run information '
        # Save flags and metric value in parameter.txt
        flag_reader.write_flags_and_BVE(self.flags,metric,self.name)

        # Save numpy arrays
        #np.savez(self.dir+'/avg_top.npz', avg_top)
        #np.savez(self.dir+'/furthest.npz', furthest)

        # Time of run
        print("Time elapsed: ", elapsed)
        with open(self.name+'/time.txt', 'w') as f:
            f.write(str(elapsed))


        ' Plot characteristics of run '
        #Scatter_Animator

    def best_validation_loss(self):
        return self.worker.best_validation_loss()

    def metric_center(self):
        ' General formula for measuring the symmetry and center of a run'
        with torch.no_grad():
            avg_top = []
            amass = [np.zeros_like(self.worker.models[0].p_list())]
            for i in range(len(self.worker.avg_tops)):
                amass += self.worker.avg_tops[i]
                avg_top.append(amass / (i + 1))

            furthest = self.worker.furthest_model

            distance_metric = []
            for i in range(len(avg_top)):
                distance_metric.append(np.sqrt(np.linalg.norm(avg_top[i]) / np.linalg.norm(furthest[i] / 2)))

            distance_metric = np.array(distance_metric)

            avg_top = np.squeeze(np.array(avg_top))
            furthest = np.squeeze(np.array(furthest))

        return distance_metric, avg_top, furthest


    def step(self):
        print("Gen: ", self.gen, "Mutation Rate: ", self.worker.mut)

        if not self.sched is None:
            rec = self.worker.mut
            self.worker.mut = self.sched.check(self.worker.best_validation_loss(), self.gen)
            if rec != self.worker.mut:
                print("At Gen: ", self.gen, " mutation power changed to ", self.worker.mut)

                with open(self.name+'/scheduler.txt', 'a') as file:
                    file.write("{"+"'Gen': {}, 'Mut': {}".format(self.gen,self.worker.mut)+"}, ")

        self.worker.run(self.gen)
        self.gen += 1