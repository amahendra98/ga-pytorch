import argparse
import flag_reader
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from ga import GA
import random

if __name__ == '__main__':
    f = flag_reader.read_flag()
    t = f.trunc_threshold
    m = f.mutation_power
    p = f.pop_size

    #('generational scheduler', 10, [(MUT,0), (0.02,20),(0.01,60)])

    f.folder = f.folder+"/T{}_M_P{}_generational_scheduler".format(t,p)
    f.device=['cuda:0']
    print(f)
    ga = GA(f)
    ga.run()