import argparse
import pickle
import os
import numpy as np

# Own module
from parameters import *

def read_flag():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, help='Device list for model storage', default = DEVICE_LIST)
    parser.add_argument('--pop-size', type=int, help='Population size.', default=POP_SIZE)
    parser.add_argument('--generations', type=int, default=GENERATIONS, metavar='N',
                            help='number of generations to train (default: 1000)')
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
    parser.add_argument('--x-lim',nargs='+', type=int, default=XLIM, help='Gif X limit')
    parser.add_argument('--y-lim', nargs='+', type=int, default=YLIM, help='Gif Y limit')
    parser.add_argument('--gif-time', type=int, default=TIME, help='Gif total time')
    flags = parser.parse_args()

    return flags

def save_flags(flags, save_dir, save_file="flags.obj"):
    with open(os.path.join(save_dir, save_file),'wb') as f:          # Open the file
        pickle.dump(flags, f)               # Use Pickle to serialize the object

def load_flags(save_dir, save_file="flags.obj"):
    with open(os.path.join(save_dir, save_file), 'rb') as f:  # Open the file
        flags = pickle.load(f)  # Use pickle to inflate the obj back to RAM
    return flags

def write_flags_and_BVE(flags, metric, save_dir):
    """
    The function that is usually executed at the end of the training where the flags and the best validation loss are recorded
    They are put in the folder that called this function and save as "parameters.txt"
    This parameter.txt is also attached to the generated email
    :param flags: The flags struct containing all the parameters
    :param best_validation_loss: The best_validation_loss recorded in a training
    :return: None
    """
    # To avoid terrible looking shape of y_range
    #yrange = flags.y_range
    # yrange_str = str(yrange[0]) + ' to ' + str(yrange[-1])
    #yrange_str = [yrange[0], yrange[-1]]
    flags_dict = vars(flags)
    flags_dict_copy = flags_dict.copy()  # in order to not corrupt the original data strucutre
    #flags_dict_copy['y_range'] = yrange_str  # Change the y range to be acceptable long string
    flags_dict_copy['best_validation_loss'] = metric  # Append the metric
    # Convert the dictionary into pandas data frame which is easier to handle with and write read
    print(flags_dict_copy)
    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        print(flags_dict_copy, file=f)
    # Pickle the obj
    save_flags(flags, save_dir=save_dir)
