import argparse
import flag_reader
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from ga import GA
import random

if __name__ == '__main__':
    f = flag_reader.read_flag()
    f.folder = 'results/sweeps/sweep_02/P{}_T{}_I{}_K{}'.format(f.pop_size, f.trunc_threshold,
                                                                     f.insertion, f.k)
    f.device=['cuda:0']
    print(f)
    ga = GA(f)
    ga.run()