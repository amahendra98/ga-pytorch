import argparse
import pickle
import os
from numpy import inf
import numpy as np

# Own module
from parameters import *

def read_flag():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, help='Device list for model storage', default = DEVICE_LIST)
    parser.add_argument('--pop-size', type=int, help='Population size.', default=POP_SIZE)
    parser.add_argument('--gen-end', type=int, default=GEN_END, metavar='N',
                            help='number of generations to train')
    parser.add_argument('--gen-start', type=int, default=GEN_START, metavar='N',
                            help='number of generations to train')
    parser.add_argument('--linear', type=list, default=LINEAR, metavar='Model layer units')
    parser.add_argument('--top', type=int, default=TOP, metavar='N',
                            help='numer of top elites that should be re-evaluated')
    parser.add_argument('--trunc-threshold', type=int, default=TRUNC,
                            help='Fraction of models that survive each generation')
    parser.add_argument('--mutation-power', type=float, help="Mutation Power", default = MUT)
    parser.add_argument('--novelty-weight', type=float, help="Factor multiplying Novelty Score", default=NOV_WEIGHT)
    parser.add_argument('--loss-weight', type=float, help="Factor multiplying negative Loss", default=LOSS_WEIGHT)
    parser.add_argument('--num-batches', type=int, help='Number of batches of trianing data', default = NUM_BATCHES)
    parser.add_argument('--insertion', type=float, help="Insertion Probability", default = INSERTION)
    parser.add_argument('--k', type=float, help="kth nearest neighbor by percent Pop", default = K_NEAREST)
    parser.add_argument('--seed', type=int, default=SEED, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--folder', type=str, default=FOLDER, metavar='N',help='folder to store results')
    parser.add_argument('--schedule-args', type=tuple, default=SCHED_ARGS, help='Arguments to run scheduler')

    parser.add_argument('--save_BVL', type=int, default=BVL, help='How often best validation loss should be saved. 0 '
                                                                  'or inf mean only save at end')
    parser.add_argument('--save_time', type=int, default=TIME, help='How often best validation loss should be saved. '
                                                                      '0 or inf mean only save at end')
    parser.add_argument('--save_models', type=int, default=MOD, help='How often best validation loss should be saved.'
                                                                       '0 means never save, inf means only save at end')
    parser.add_argument('--safe-mutation', type=str, default=SAFE_MUT, help='Type of safe mutation to run i.e. regular,'
                                                                            'SM-G, SM-G-SO')

    flags = parser.parse_args()
    return flags

def write_flags_to_file(flags, save_dir):
    ' Create a parameters.txt file that holds critical run infromation'
    flags_dict = vars(flags)
    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        print(flags_dict, file=f)
    save_flags(flags, save_dir=save_dir)


# TODO: Consider caching each add_to_file call and forgoing the file read step
def add_to_file(save_dir, BVL=None, time=None):
    ' Store best validation loss and/or time so far in parameters.txt under best_validation_loss and time'

    path = os.path.join(save_dir, 'parameters.txt')

    with open(path, 'r') as f:
        flags_dict = eval(f.read())

    with open(path, 'w') as f:
        if not BVL == None:
            flags_dict['best_validation_loss'] = BVL

        if not time == None:
            flags_dict['time'] = time

        print(flags_dict, file=f)


def save_flags(flags, save_dir, save_file="flags.obj"):
    ' Save flags as pickle-able object'
    with open(os.path.join(save_dir, save_file),'wb') as f:          # Open the file
        pickle.dump(flags, f)               # Use Pickle to serialize the object

def load_flags(save_dir, save_file="flags.obj"):
    ' Load flags.obj (pickle-able flags object) '
    with open(os.path.join(save_dir, save_file), 'rb') as f:  # Open the file
        flags = pickle.load(f)  # Use pickle to inflate the obj back to RAM
    return flags