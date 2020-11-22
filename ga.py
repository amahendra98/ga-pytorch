import torch
import numpy as np
import time
import os

# Own Modules
import flag_reader
from schedulers import *

class GA:
    def __init__(self, flags):
        if flags.gen_end == None:
            flags.gen_end = float('inf')
        self.max_gen = flags.gen_end
        self.start_gen = flags.gen_start

        self.start_time = 0
        devices = flags.device[0]
        pop_size = flags.pop_size
        # TODO: Change name variable to a more informative title
        self.name = flags.folder
        self.gen = 0
        self.sched = None
        self.save_mod = flags.save_models
        self.save_BVL = flags.save_BVL
        self.save_time = flags.save_time

        if self.save_mod != 0:
            self.save = self.save_GA_and_models
        else:
            self.save = self.save_GA


        # TODO: Wrap scheduler selection in a function or import scheduler object from another file that can change to
        # be one of these schedulers
        if flags.schedule_args == None:
            self.sched = None
        else:
            func_args = flags.schedule_args[1:]
            if flags.schedule_args[0] == 'generational_scheduler':
                self.sched = generational_scheduler(*func_args)
            if flags.schedule_args[0] == 'value_based_scheduler':
                self.sched = value_based_scheduler(*func_args)
            if flags.schedule_args[0] == 'step_length_doubling_scheduler':
                self.sched = step_length_doubling_scheduler(*func_args)
            if flags.schedule_args[0] == 'variable_length_value_scheduler':
                self.sched = variable_length_value_scheduler(*func_args)

        # TODO: Delete this population splitting stuff for now, multi-GPU maybe useful in future
        ' Divide population amongst GPUs and store how many models on each GPU '
        pop_per_device = int( pop_size/len(devices ))
        remainder = int( pop_size%len(devices) )
        self.pop_array = [pop_per_device]*len(devices)
        self.pop_array[1:remainder] += [1]*remainder

        ' Initialize GPUWorkers'
        from train import GPUWorker
        self.devices = devices #Creating single gpu version first
        self.worker = GPUWorker(flags)

        ' Save Flags to parameters.txt file (identifies run) '
        print("Storing run info to ", self.name)
        flag_reader.write_flags_to_file(flags, self.name)

    def run(self):
        ' DO GA ALGORITHM '

        # TODO: Hide this memory management in a function in train.py
        ' Clear the space occupied by rmem (space was occupied to force pytorch to allocate enough memory for run '
        self.worker.rmem = None

        ' Start timing the run '
        self.start_time = time.time()

        ' Execute the run loop '
        gen = self.start_gen
        while gen < self.max_gen:
            gen = self.step(gen)

        ' Functions to execute post-run '
        self.save()


    def best_validation_loss(self):
        return self.worker.best_validation_loss()

    def get_models(self):
        return self.worker.get_models()


    def step(self, gen):
        print("Gen: ", gen)
        # Calculate mutation rate from scheduler
        # TODO: Wrap this function into the scheduler wrapper class
        if not self.sched is None:
            rec = self.worker.mut
            self.worker.mut = self.sched.check(self.worker.best_validation_loss(), gen)
            if rec != self.worker.mut:
                print("At Gen: ", gen, " mutation power changed to ", self.worker.mut)

                with open(self.name+'/scheduler.txt', 'a') as file:
                    file.write("{"+"'Gen': {}, 'Mut': {}".format(gen,self.worker.mut)+"}, ")

        # Save necessary info
        self.save(gen)

        # Run GA on each worker and augment generation
        self.worker.run(gen)
        return gen + 1

    def save_GA(self,gen=0):
        ' Save best validation loss and time elapsed to parameters.txt'
        empty = True
        if gen % self.save_BVL == 0:
            BVL = float(self.best_validation_loss().cpu())
            empty = False
        if gen % self.save_time == 0:
            elapsed = time.time() - self.start_time
            empty = False
        if empty == True:
            return
        else:
            flag_reader.add_to_file(self.name,BVL,elapsed)

    def save_GA_and_models(self, gen=0):
        ''' Save best validation loss and time elapsed to parameters.txt,
        and then save all models produced to final_population.pt file'''
        self.save_GA(gen)

        if gen % self.save_mod == 0:
            m_dict = {}
            models = self.get_models()
            for i, model in enumerate(models):
                m_dict['model{}_state_dict'.format(i)] = model.state_dict()
            torch.save(m_dict, self.name + '/final_population.pt')

    # TODO: Load GA from saved information
    def load_GA(self):
        pass

    # TODO: Hide metrics in another file and import this function as needed
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