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
from train import DataStore

from multiprocessing import set_start_method
set_start_method('spawn', force=True)


class GA:
    def __init__(self, elite_evals, top, threads, timelimit, pop_size, setting, devices, num_batches):
        '''
        Constructor. 
        '''
        self.top  = top  #Number of top individuals that should be reevaluated
        self.elite_evals = elite_evals  #Number of times should the top individuals be evaluate
        self.pop_size = pop_size
        self.threads = threads
        multi_process = threads>1
        self.truncation_threshold = int(pop_size/2)  #Should be dividable by two
        self.devices = devices

        from train import GAIndividual

        ' Calculate population per device '
        num_dev = len(devices)

        remainder = pop_size % num_dev
        common = pop_size / num_dev
        pop_tracker = [common] * num_dev
        pop_tracker[0:remainder] += [1]*remainder

        ' Load and datastores into an array, one loaded per device '
        self.datastores = [DataStore(devices[i],pop_tracker[i],num_batches) for i in range(num_dev)]

        ' Store one GAIndividual (stores a model and handles its training) for each in population '
        self.P = []
        for i in range(pop_size):
            dev_idx = i % num_dev
            dev = devices[dev_idx]
            self.P.append( GAIndividual(dev, timelimit, setting, self.datastores[dev_idx],multi=multi_process) )

        
    def run(self, max_generations, filename, folder):
        ' Handles Actual Genetic Algorithm'

        Q = []

        ' Assume lowest possible training fitness value '
        max_train_fitness = -sys.maxsize
        max_test_fitness = -sys.maxsize

        fitness_file = open(folder+"/fitness_"+filename+".txt", 'a')

        ind_fitness_file = open(folder+"/individual_fitness_"+filename+".txt", 'a')

        ' Setup Graphing File '
        graphing_file = open(folder + "/fitness_" + filename + ".csv", 'a')
        graphing_file.write("Generation, avg_train_fit, avg_test_fit, max_train_fit_gen, max_test_fit_gen," \
                            "max_train_fit, max_test_fit, time elapsed for gen\n")

        i = 0
        P = self.P

        pop_name = folder+"/pop_"+filename+".p"

        ' Code to load previously saved populations '
        if exists( pop_name ):
            pop_tmp = torch.load(pop_name)

            print("Loading existing population ",pop_name, len(pop_tmp))

            idx = 0
            for s in pop_tmp:
                 P[idx].r_gen.lorentz.load_state_dict ( s['lorentz'].copy() )
                 i = s['generation'] + 1
                 idx+=1
                 
        ' Training Loop '
        while (True):
            pool = multiprocessing.Pool(self.threads)

            start_time = time.time()

            print("Generation ", i)
            sys.stdout.flush()


            ' First trains models '
            print("Evaluating individuals on training set: ",len(P) )
            for s in P:
                ' Run adds task to async pool '
                s.run_solution(pool, 1, force_eval=True)

            train_fitness = []
            test_fitness = []

            for s in P:
                s.is_elite = False
                ' evaluate_solution grabs task from async pool '
                train_f, test_f = s.evaluate_solution(1)
                train_fitness += [train_f]
                test_fitness += [test_f]

            ' Then sorts models '
            self.sort_objective(P)

            max_train_fitness_gen = -sys.maxsize #keep track of highest fitness this generation
            max_test_fitness_gen = -sys.maxsize

            ' Evaluates elites '
            # Unnecessary for this code
            print("Evaluating elites on training set: ", self.top)
            #for k in range(self.top):
            #    P[k].run_solution(pool, self.elite_evals)

            for k in range(self.top):
                #train_f, test_f = P[k].evaluate_solution(self.elite_evals)

                ' Determine elites based on if fitness is > max_train_fitness of last iteration '
                if -P[k].fitness > max_train_fitness_gen:   #All logic and evaluation is done with train data
                    'Negative values important for sorting somehow'
                    max_train_fitness_gen = -P[k].fitness
                    max_test_fitness_gen = -P[k].testing_fitness
                    elite = P[k]

                ' Determine champion by comparing if this is the best fitness ever found '
                if -P[k].fitness > max_train_fitness: #best fitness ever found
                    max_train_fitness = -P[k].fitness
                    max_test_fitness = -P[k].testing_fitness
                    print("\tFound new champion! Train fitness: ", max_train_fitness," Test fitness: ", max_test_fitness)

                    best_ever = P[k]
                    sys.stdout.flush()
                    
                    torch.save({'lorentz': elite.r_gen.lorentz.state_dict(), 'fitness':train_f}, "{0}/best_{1}G{2}.p".format(folder, filename, i))

            elite.is_elite = True  #The best 

            sys.stdout.flush()

            ' Close pool once all tasks are completed - synchronization step '
            pool.close()

            ' Delete the worst models and keep the best, threshold is truncation threshold '
            Q = []
            replaced = []
            if len(P) > self.truncation_threshold-1:
                for j in range(len(P)-1,self.truncation_threshold - 2,-1):
                    tmp = P.pop(j)
                    replaced.append((tmp.device, tmp.r_gen.dataSource))
                    ' replaced holds data necessary for new model that will eventually take pooped' \
                    'models place to run in popped models GAIndividual context'

            P.append(elite) #Maybe it's in there twice now but that's okay

            save_pop = []

            for s in P:
                 ind_fitness_file.write( "Gen\t%d\tFitness\t%f\n" % (i, -s.fitness )  )  
                 ind_fitness_file.flush()

                 save_pop += [{'lorentz': s.r_gen.lorentz.state_dict(), 'train_fitness':train_fitness,
                               'test_fitness':test_fitness, 'generation':i}]
                 

            if (i % 25 == 0):
                print("saving population")
                torch.save(save_pop, folder+"/pop_"+filename+".p")
                print("done")

            print("Creating new population ...", len(P))
            Q = self.make_new_pop(P, replaced)

            P.extend(Q)

            elapsed_time = time.time() - start_time

            print( "%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\tTime\t%f\n" % (i, np.mean(train_fitness),
                                                                           max_train_fitness_gen, max_train_fitness,
                                                                           elapsed_time) )  # python will convert \n to os.linesep

            fitness_file.write( "%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\tTime\t%f\n" % (i, np.mean(train_fitness),
                                                                                        max_train_fitness_gen,
                                                                                        max_train_fitness,
                                                                                        elapsed_time) )  # python will convert \n to os.linesep
            fitness_file.flush()
            graphing_file.write("%d,%f,%f,%f,%f,%f,%f,%f\n" % (i, np.mean(train_fitness), np.mean(test_fitness),max_train_fitness_gen,
                                                      max_test_fitness_gen,max_train_fitness,max_test_fitness, elapsed_time))
            graphing_file.flush()

            if (i > max_generations):
                break

            gc.collect()

            i += 1

        print("Testing best ever: ")
        pool = multiprocessing.Pool(self.threads)

        best_ever.run_solution(pool, 100, force_eval = True)
        train_f, test_f = best_ever.evaluate_solution(100)
        print(train_f, test_f)
        
        fitness_file.write( "Train fitness\t%f\tTest fitness\t%f\n" % (train_f, test_f) )
        fitness_file.close()
        ind_fitness_file.close()
        graphing_file.close()

                                
    def sort_objective(self, P):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if s1.fitness > s2.fitness:
                    P[j - 1] = s2
                    P[j] = s1
                    

    def make_new_pop(self, P, replaced):
        '''
        Make new population Q, offspring of P. 
        '''
        Q = []
        
        while len(Q) < self.truncation_threshold:
            selected_solution = None
            idx = 0

            ' Choose a remaining model from the population '
            s1 = random.choice(P)
            s2 = s1
            while s1 == s2:
                s2 = random.choice(P)

            if s1.fitness < s2.fitness: #Lower is better
                selected_solution = s1
            else:
                selected_solution  = s2

            if s1.is_elite:  #If they are the elite they definitely win
                selected_solution = s1
            elif s2.is_elite:  
                selected_solution = s2

            ' Clone child solution in and mutate '
            child_solution = selected_solution.clone_individual(replaced[idx])
            child_solution.mutate()

            if (not child_solution in Q):    
                Q.append(child_solution)
                idx += 1
        
        return Q
        
