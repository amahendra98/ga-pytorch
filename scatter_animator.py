import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
import pandas as pd
from flag_reader import *
import plotly.express as px
import os

class Data_Walker():
    def __init__(self, dir, thresh, skip=None, end=None):
        self.skip = skip
        self.end = end
        self.dir = dir
        self.thresh = thresh
        self.Pop = [2000,1500,1000, 500]#[50, 100, 200, 500]
        self.Trunc = [0.1,0.2,0.4,0.5]#[0.02, 0.05,0.1,0.15,0.2]
        self.Mut = [0.02,0.01,0.001,0.1,0.5,1]
        self.Insertion = [0.05, 0.3, 0.45] #[0.01,0.05,0.1,0.2,0.5]
        self.K_Nearest = [0.1, 0.05, 0.005, 0.025] #self.Trunc

    def exp_walk2(self, func_string):
        pre_frame = []

        for root,dirs,files in os.walk(self.dir):
            for file in files:
                if file.find(".npz") != -1:
                    if file.find("furthest")!=-1 or file.find("avg_top")!=-1:
                        continue
                    print(root)
                    print(file)
                    os.remove(root + "/" + file)
            '''
             for folder in dirs:
                if folder == "imgs":
                    continue
                if func_string.find('e') != -1:
                    self.count_successful_gen('/'+folder)
                if func_string.find('h') != -1:
                    d_row = self.add_pc_data('/'+folder)
                    pre_frame.append(d_row)
            '''
        return pre_frame

        """
            for file in files:
            if file.find(".npz")!=-1:
                print(root)
                print(file)
                os.remove(root+"/"+file)

        """

    def exp_walk1(self, func_string):
        pre_frame =[]

        for p in self.Pop:
            folder = '/P{}_T{}_I{}_K{}'.format(p, TRUNC, INSERTION, K_NEAREST)
            if func_string.find('e') != -1:
                self.count_successful_gen(folder)
            if func_string.find('h') != -1:
                d_row = self.add_pc_data(folder)
                pre_frame.append(d_row)

        for t in self.Trunc:
            folder = '/P{}_T{}_I{}_K{}'.format(POP_SIZE, t, INSERTION, K_NEAREST)
            if func_string.find('e') != -1:
                self.count_successful_gen(folder)
            if func_string.find('h') != -1:
                d_row = self.add_pc_data(folder)
                pre_frame.append(d_row)

        for i in self.Insertion:
            folder = '/P{}_T{}_I{}_K{}'.format(POP_SIZE, TRUNC, i, K_NEAREST)
            if func_string.find('e') != -1:
                self.count_successful_gen(folder)
            if func_string.find('h') != -1:
                d_row = self.add_pc_data(folder)
                pre_frame.append(d_row)

        for k in self.K_Nearest:
            folder = '/P{}_T{}_I{}_K{}'.format(POP_SIZE, TRUNC, INSERTION, k)
            if func_string.find('e') != -1:
                self.count_successful_gen(folder)
            if func_string.find('h') != -1:
                d_row = self.add_pc_data(folder)
                pre_frame.append(d_row)

        return pre_frame

    def exp_walk(self, func_string):
        count = 0
        pre_frame = []

        for p in self.Pop:
            for t in self.Trunc:
                for m in self.Mut:
                    for i in self.Insertion:
                        for k in self.K_Nearest:
                            count += 1
                            if count in self.skip:
                                continue

                            if count == self.end:
                                break

                            folder = "/P{}_T{}_M{}_I{}_K{}".format(p,t,m,i,k)

                            if func_string.find('e') != -1:
                                self.count_successful_gen(folder)
                            if func_string.find('h') != -1:
                                d_row = self.add_pc_data(folder)
                                pre_frame.append(d_row)

        return pre_frame


    def count_successful_gen(self, folder):
        avg_top = np.load(self.dir+folder+"/avg_top.npz")['arr_0']
        furthest = np.load(self.dir+folder+"/furthest.npz")["arr_0"]

        distance_metric = []
        for i in range(len(avg_top)):
            distance_metric.append(np.sqrt(np.linalg.norm(avg_top[i]) / np.linalg.norm(furthest[i] / 2)))

        d_ratio = np.array(distance_metric)
        bool_count = d_ratio < self.thresh
        gen_count = np.sum(bool_count)

        with open(self.dir+folder+'/suc_gen.txt','w') as f:
            f.truncate()
            f.write(str(gen_count))

    def add_pc_data(self, folder):
        row = None
        with open(self.dir+folder+'/parameters.txt','r') as f:
            content = f.read()
            p_dict = eval(content)
            #row = [p_dict['pop_size'], p_dict['trunc_threshold'],p_dict['mutation_power'],p_dict['insertion'],p_dict['k']]
            row = [p_dict['pop_size'], p_dict['trunc_threshold'], p_dict['insertion'],p_dict['k']]

        with open(self.dir+folder+'/suc_gen.txt','r') as f:
            row.append(int(f.readline()))

        return(row)


    def plot_parallel_coordinates(self, func_string):
        preFrame = self.exp_walk2(func_string)

        #cols = ['Pop', 'Truncation Ratio', 'Mutation', 'Insertion', 'K-nearest neighbor ratio', 'Metric']
        cols = ['Pop', 'Truncation Ratio', 'Insertion', 'K-nearest neighbor ratio', 'Metric']
        dFrame = pd.DataFrame(np.array(preFrame), columns=cols)

        lw_bound = 0
        dplot = dFrame[dFrame.Metric.between(lw_bound, 200)]
        fig = px.parallel_coordinates(dplot, color='Metric', dimensions=cols[0:4],
                                      title="Plot with threshold at {}, restricted to metrics above {}".format(
                                          self.thresh, lw_bound))
        fig.show()

        '''
        for i in range(lw_bound,0,-1):
            dplot = dFrame[dFrame.Metric.between(i,483)]
            fig = px.parallel_coordinates(dplot, color='Metric', dimensions=cols[0:5],
                                title="Plot with threshold at {}, restricted to metrics above {}".format(self.thresh,i))
            fig.write_image(self.dir+'/imgs/fig_{}.jpg'.format(i))
        '''

class Scatter_Animator():
    def __init__(self, name, total_time,x_lim,y_lim):
        super(Scatter_Animator, self).__init__()
        self.name = name
        self.total_time = total_time
        self.x_lim = y_lim
        self.y_lim = x_lim

    def animate(self):
        ' Create 2d Parameter Scatter Animation '

        # Load stored data
        arcv_mod = np.load(self.name+"/arcv_mod.npz")["arr_0"]
        pop = np.load(self.name+"/pop.npz")["arr_0"]
        elite_idx = np.load(self.name+"/elite_idx.npz")["arr_0"]
        arcv_term_idx = np.load(self.name+"/arcv_term_idx.npz")["arr_0"]
        avg_top = np.load(self.name+"/avg_top.npz")['arr_0']
        furthest = np.load(self.name+"/furthest.npz")["arr_0"]

        avg_top = np.squeeze(np.array(avg_top))
        furthest = np.squeeze(np.array(furthest))

        '''
        mutates = []
        for gen in range(len(pop)):
            mutates.append([])
            for el in range(len(elite_idx[0])):
                mutates[gen].append(np.load(self.name+"/mutates/gen_{}_elite_{}.npz".format(gen,el))["arr_0"])
        mutates = np.array(mutates)
        '''

        # Setting Up master frame array
        master = np.empty((1, 2))
        colormap = np.array([(0, 0, 1, 1), (1, 0, 0, 0.5), (0, 1, 0, 1), (1, 1, 1, 1), (1, 1, 0, 1), (0,1,1,1)])
        m_c = np.array([0], dtype=np.int32)
        master_frame_len = [0]

        for gen in range(len(pop)):
            # Frame 1: master = array of pop (blue) followed by archive (red)
            gen_pop = pop[gen]
            gen_arcv = arcv_mod[0:arcv_term_idx[gen]]
            champions = np.empty((1, 2))

            # Add population in this generation to master
            master = np.vstack((master, gen_pop))
            m_c = np.concatenate((m_c, np.array([0] * len(gen_pop), dtype=np.int32)))
            # Add archived models from 0 up until this generation to master
            if np.any(gen_arcv):
                master = np.vstack((master, gen_arcv))
                m_c = np.concatenate((m_c, np.array([1] * arcv_term_idx[gen], dtype=np.int32)))

            if gen == 0:
                master = np.delete(master, 0, 0)
                m_c = np.delete(m_c, 0, 0)
            else:
                for g in range(gen):
                    champ = pop[g][elite_idx[g][0]]
                    champions = np.vstack((champions, champ))
                champions = np.delete(champions, 0, 0)
                master = np.vstack((master, champions))
                m_c = np.concatenate((m_c, np.array([4] * len(champions), dtype=np.int32)))

            master = np.vstack((master,avg_top[0:gen]))
            master = np.vstack((master,furthest[gen]))
            m_c = np.concatenate((m_c, np.array([3] * gen + [5], dtype=np.int32)))

            master_frame_len.append(len(master))

            # Frame 2: Master = array of truncation survivors (blue) followed by archive (red)
            gen_elite = pop[gen][elite_idx[gen][0:len(elite_idx[gen])]]

            master = np.vstack((master, gen_elite))
            m_c = np.concatenate((m_c, np.array([2] * len(gen_elite), dtype=np.int32)))

            if np.any(gen_arcv):
                master = np.vstack((master, gen_arcv))
                m_c = np.concatenate((m_c, np.array([1] * arcv_term_idx[gen], dtype=np.int32)))

            if gen > 0:
                master = np.vstack((master, champions))
                m_c = np.concatenate((m_c, np.array([4] * len(champions), dtype=np.int32)))

            master = np.vstack((master,avg_top[0:gen]))
            m_c = np.concatenate((m_c, np.array([3] * gen, dtype=np.int32)))

            master_frame_len.append(len(master))

        '''
            # Frames 3 -> 3 + trunc_threshold: Master = array of top, mutations, and archive
            for i in range(len(elite_idx[0])):
                hi_lite = gen_elite[1:i + 1]
                muts = mutates[gen][i]

                lited = np.empty((1, 2))
                for m in mutates[gen][0:i]:
                    if not np.any(m):
                        continue
                    z = np.array(m)
                    lited = np.vstack((lited, z))
                lited = np.delete(lited, 0, 0)

                not_lit = gen_elite[i + 1:len(gen_elite)]

                champions = np.empty((1, 2))
                for g in range(gen + 1):
                    champ = pop[g][elite_idx[g][0]]
                    champions = np.vstack((champions, champ))
                champions = np.delete(champions, 0, 0)

                master = np.vstack((master, lited))
                master = np.vstack((master, not_lit))
                m_c = np.concatenate((m_c, np.array([0] * len(lited) + [0] * len(not_lit), dtype=np.int32)))

                if np.any(gen_arcv):
                    master = np.vstack((master, gen_arcv))
                    m_c = np.concatenate((m_c, np.array([1] * arcv_term_idx[gen], dtype=np.int32)))

                master = np.vstack((master, hi_lite))
                m_c = np.concatenate((m_c, np.array([2] * len(hi_lite), dtype=np.int32)))

                if np.any(muts):
                    master = np.vstack((master, muts))
                    m_c = np.concatenate((m_c, np.array([3] * len(muts), dtype=np.int32)))

                master = np.vstack((master, champions))
                m_c = np.concatenate((m_c, np.array([4] * len(champions), dtype=np.int32)))

                master_frame_len.append(len(master))
        '''

        # Initial Figure frame
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(xlim=self.x_lim, ylim=self.y_lim)
        scat = ax.scatter(master[0:master_frame_len[1]][:, 0], master[0:master_frame_len[1]][:, 1],
                          c=colormap[m_c[0:master_frame_len[1]]])
        gen_text = ax.text(self.x_lim[0]+0.1, self.y_lim[1]-0.2, 'gen 0')
        gen_text.set_backgroundcolor((1, 1, 1))
        ax.set_facecolor((0, 0, 0))
        plt.xlabel("p1")
        plt.ylabel("p2")


        # Update Function
        def animate_scatter(i):
            step = ['Population', 'Truncation'] + ['Mutation'] * len(elite_idx[0]) # = truncation threshold
            modulus = i % 2#(2 + len(elite_idx[0]))
            gen = int(i / 2)#(2 + len(elite_idx[0])))

            scat.set_offsets(master[master_frame_len[i]:master_frame_len[i + 1]])
            scat.set_color(colormap[m_c[master_frame_len[i]:master_frame_len[i + 1]]])
            gen_text.set_text('Gen {}: {}'.format(gen, step[modulus]))


        # Animation Writer
        num_frames = len(pop) * 2 #(2 + len(elite_idx[0]))
        Scatter_animation = FuncAnimation(fig, animate_scatter, interval=1, frames=num_frames)
                                          #frames=len(pop) * (2 + len(elite_idx[0])))

        fps = num_frames/self.total_time
        Scatter_animation.save("{}/Run.gif".format(self.name, self.name), fps=fps, writer='Pillow')

if __name__ == '__main__':
    name = './ga-pytorch/results/sweeps/sweep_02'
    if len(sys.argv) > 1:
        func = sys.argv[1]
    if len(sys.argv) > 2:
        name = sys.argv[2]

    print("Function: ",func,"\tDirectory: ",name," is being used")

    thresh = 0.4
    skip = None
    end = None

    if func =='a':
        total_time = 5
        x_lim = (-10, 10)
        y_lim = (-10, 10)
        s = Scatter_Animator(name, total_time, x_lim, y_lim)
        s.animate()
    if func =='e':
        d = Data_Walker(name, thresh, skip, end)
        d.exp_walk2(func)
    if func =='h' or func=='eh':
        d = Data_Walker(name, thresh, skip, end)
        d.plot_parallel_coordinates(func)