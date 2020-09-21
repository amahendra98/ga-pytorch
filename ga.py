import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.path as path
import numpy as np
import time

class GA:
    def __init__(self, top, pop_size, devices, trunc, batches, mut,maxgen,lw,nw,p):
        '''
        Constructor loads parameters, loads data and models onto each device
        '''
        self.model_flags = {'linear' : [2, 100, 100], 'num_spec_point' : 300, 'num_lorentz_osc' : 1,
                            'freq_low' : 0.5,'freq_high' : 5}
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
        self.workers = [GPUWorker(dev,self.pop_array[i],batches,top,trunc, mut, self.model_flags,maxgen,lw,nw,p)
                        for i, dev in enumerate(devices)]


    def run(self, filename, folder):

        ' Genetic Algorithm starts running here '

        'Ideally get rid of while loop and turn it into recursive gpu calls without cpu interference'
        i = 0

        start = time.time()
        while(i<self.max_gen):
            for w in self.workers: w.run(i)
            i += 1
            print("gen: ",i)
        end = time.time()

        print("Time Elapsed: ", end - start)

        fig = plt.figure()
        lines = []
        for i in range(1,5):
            ax = fig.add_subplot(1, 4, i)
            ax.set(xlim=(0,5),ylim=(0,self.workers[0].lorentz_plot[(i*4) + 3].max() + 1))
            lines.append(ax.plot(np.linspace(0.5, 5, 300), self.workers[0].lorentz_plot[(i*4)], color='tab:blue',label='Champion')[0])
            lines.append(ax.plot(np.linspace(0.5, 5, 300), self.workers[0].lorentz_plot[(i*4) + 1], color='tab:purple',label='worst elite')[0])
            lines.append(ax.plot(np.linspace(0.5, 5, 300), self.workers[0].lorentz_plot[(i*4) + 2], color='tab:red', label='worst')[0])
            lines.append(ax.plot(np.linspace(0.5, 5, 300), self.workers[0].lorentz_plot[(i*4) + 3], color='tab:orange', label='Truth Spectra')[0])
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("e2")


        n,bins = np.histogram(self.workers[0].hist_plot[0],bins=np.arange(0,5,0.05))

        left = np.array(bins[:-1])
        right = np.array(bins[1:])
        bottom = np.zeros(len(left))
        top = bottom + n
        nrects = len(left)

        nverts = nrects * (1 + 3 + 1)
        verts = np.zeros((nverts, 2))
        codes = np.ones(nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        verts[0::5, 0] = left
        verts[0::5, 1] = bottom
        verts[1::5, 0] = left
        verts[1::5, 1] = top
        verts[2::5, 0] = right
        verts[2::5, 1] = top
        verts[3::5, 0] = right
        verts[3::5, 1] = bottom

        def animate_lorentz(i):
            for j, line in enumerate(lines):
                line.set_ydata(self.workers[0].lorentz_plot[j + (i*16)])

        Lorentz_animation = FuncAnimation(fig, animate_lorentz, interval=200, frames=100)
        Lorentz_animation.save('Lorentz_{}.gif'.format(start), writer='imagemagick')


        patch = None
        def animate_histogram(i):
            n, bins = np.histogram(self.workers[0].hist_plot[i], bins=np.arange(0,5,0.05))
            top = bottom + n
            verts[1::5, 1] = top
            verts[2::5, 1] = top
            return [patch, ]

        fig, ax = plt.subplots()
        barpath = path.Path(verts, codes)
        patch = patches.PathPatch(
            barpath, facecolor='blue', edgecolor='purple', alpha=0.5)
        ax.add_patch(patch)

        ax.set_xlim(left[0], right[-1])
        #ax.set_xticks([0, 0.1, 0.3, 0.5, 0.8, 1, 2, 5, 10, 20, 50])
        #ax.set_xscale('log')
        ax.set_ylim(bottom.min(), 2*top.max())
        ax.set_xlabel("MSE_Loss")
        ax.set_ylabel("Number of datasets")

        Histogram_animation = FuncAnimation(fig, animate_histogram, 100, repeat=False, blit=True)
        Histogram_animation.save('Histogram_{}.gif'.format(start), writer='imagemagic')

        plt.draw()


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