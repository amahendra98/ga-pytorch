import flag_reader
import torch
import numpy as np
import time

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
        self.worker = GPUWorker(flags)

    def run(self):
        ' DO GA ALGORITHM '

        ' Clear the space occupied by rmem (space was occupied to force pytorch to allocate enough memory for run'
        self.worker.rmem = None

        ' Time and execute run over max generations'
        start = time.time()
        print(self.name)
        for gz in range(self.max_gen):
            self.worker.run(gz)
            print("Gen: ", gz)


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