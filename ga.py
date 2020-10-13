import flag_reader
import torch
import numpy as np
import time
from scatter_animator import Scatter_Animator

class GA:
    def __init__(self, flags):
        self.flags = flags
        self.max_gen = flags.generations
        devices = flags.device[0]
        pop_size = flags.pop_size
        self.time = flags.gif_time
        self.x_lim = flags.x_lim
        self.y_lim = flags.y_lim
        self.name = flags.folder

        ' Divide population amongst GPUs and store how many models on each GPU '
        pop_per_device = int( pop_size/len(devices ))
        remainder = int( pop_size%len(devices) )
        self.pop_array = [pop_per_device]*len(devices)
        self.pop_array[1:remainder] += [1]*remainder

        ' Initialize GPUWorkers'
        from train import GPUWorker

        self.devices = devices #Creating single gpu version first
        self.worker = GPUWorker(devices,pop_size,flags.num_batches,flags.top,flags.trunc_threshold,flags.mutation_power, self.max_gen,
                                  flags.loss_weight,flags.novelty_weight,flags.insertion,flags.k,flags.folder)

    def run(self):
        self.worker.rmem = None
        gz = 0
        start = time.time()
        while(gz<self.max_gen):
            self.worker.run(gz)
            print("Gen: ", gz,"\t arcv length: ",self.worker.arcv_idx)
            gz += 1
        end = time.time()
        print("Time elapsed: ", end-start)

        pop = np.array(self.worker.mp_gen)
        elite_idx = np.array(self.worker.mpe_gen_idx)
        arcv_mod = np.array(self.worker.arcv_mod)
        arcv_term_idx = np.array(self.worker.arcv_term_idx)

        with torch.no_grad():
            avg_top = []
            amass = [np.zeros_like(self.worker.models[0].p_list())]
            for i in range(len(self.worker.avg_tops)):
                amass += self.worker.avg_tops[i]
                avg_top.append(amass/(i+1))

            furthest = self.worker.furthest_model

            distance_metric = []
            for i in range(len(avg_top)):
                distance_metric.append(np.sqrt(np.linalg.norm(avg_top[i])/np.linalg.norm(furthest[i]/2)))

            distance_metric = np.array(distance_metric)

            avg_top = np.squeeze(np.array(avg_top))
            furthest = np.squeeze(np.array(furthest))

            #np.savez(self.name+"/pop", pop)
            #np.savez(self.name+"/elite_idx",elite_idx)
            #np.savez(self.name+"/arcv_mod",arcv_mod)
            #np.savez(self.name+"/arcv_term_idx", arcv_term_idx)
            np.savez(self.name+"/avg_top", avg_top)
            np.savez(self.name+"/furthest", furthest)

            """
            for j in range(len(self.worker.mut_gen_idx)):
                d = self.name+"/mutates/gen_{}".format(j)
                for l in range(len(self.worker.mut_gen_idx[j])):
                    dir = d + "_elite_{}".format(l)
                    np.savez(dir,self.worker.mut_gen_idx[j][l])
            """

            flag_reader.write_flags_and_BVE(self.flags,distance_metric,self.name)

            #s = Scatter_Animator(name,self.time,self.x_lim,self.y_lim)
            #s.animate()


'''
        g_file = open(folder + "/fitness_" + filename + ".csv", 'a')
        g_file.close()

        g_file = open(folder + "/fitness_" + filename + ".csv", "r+")
        g_file.truncate(0)
        g_file.close()

        g_file = open(folder + "/fitness_" + filename + ".csv", 'a')
        g_file.write("Generation, elite_fitness\n")

        for w in self.workers:
            #arr = w.elite_vals.cpu().numpy()[0]
            arr = w.elite_vals
            for i,val in enumerate(arr):
                g_file.write("{}, {}\n".format(i,val))

        g_file.close()
'''