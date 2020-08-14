import argparse
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
import torch
import numpy as np
import train
from ga import GA
import random

import multiprocessing

#To run on headless:
#xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 main.py

torch.set_num_threads(1)

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--mutation-power', type=float, help="Mutation Power", default = 0.02)

    parser.add_argument('--num-batches', type=int, help='Number of batches of trianing data', default = 1)

    parser.add_argument('--device', type=str, help='Device list for model storage', default = ['cuda:1'])

    parser.add_argument('--pop-size', type=int, help='Population size.', default = 50)

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')

    parser.add_argument('--generations', type=int, default=20, metavar='N',
                            help='number of generations to train (default: 1000)')

    parser.add_argument('--threads', type=int, default=10, metavar='N',
                            help='threads')

    parser.add_argument('--setting', type=int, default=1, metavar='N',
                            help='0 = standard deep NE, 1 = mutate one component at a time')

    parser.add_argument('--test', type=str, default='', metavar='N',
                            help='0 = no protection, 1 = protection')

    parser.add_argument('--folder', type=str, default='results', metavar='N',
                            help='folder to store results')

    parser.add_argument('--top', type=int, default=3, metavar='N',
                            help='numer of top elites that should be re-evaluated')
    
    parser.add_argument('--elite_evals', type=int, default=20, metavar='N',
                            help='how many times should the elite be evaluated')                        

    parser.add_argument('--timelimit', type=int, default=1000, metavar='N',
                            help='time limit per evaluation')    

    parser.add_argument('--discrete', type=int, default=0, metavar='N',
                            help='discrete VAE (default 0)')

    parser.add_argument('--trunc-threshold', type=int, default=.2,
                            help='Fraction of models that survive each generation')

    args = parser.parse_args()

    " Force population size even "
    if (args.pop_size % 2 != 0):
        print("Error: Population size needs to be an even number.")
        exit()

    #The track generation process might use another random seed, so even with same random seed here results can be different
    " Set Seed "
    random.seed(args.seed)

    if not exists(args.folder):
        mkdir(args.folder)

    ' GA Class handles genetic algorithm, initialize with appropriate parameters '
    ga = GA(args.elite_evals, args.top, args.threads, args.timelimit, args.pop_size, args.setting, args.device,
            args.trunc_threshold, args.num_batches, args.mutation_power, args.generations)

    ' Do training '
    ga.run("{0}_".format(args.seed), args.folder ) #pop_size, num_gens

if __name__ == '__main__':

    main(sys.argv)
