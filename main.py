import argparse
import flag_reader
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
from ga import GA
import random

if __name__ == '__main__':
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = 0.5

    f = flag_reader.read_flag()
    t = f.trunc_threshold
    m = f.mutation_power
    p = f.pop_size
    f.mutation_power = name
    f.folder = f.folder+"/Safe-Mutation-Testing"

    dirs = listdir(f.folder)
    trial = 0
    for dir in dirs:
        if dir.count("SMG_v0_P100_M{}_".format(name)) > 0:
            trial += 1
    f.folder = f.folder + "/SMG_v0_P100_M{}_{}".format(name,trial)

    #f.device=['cuda:1']
    print(f)
    ga = GA(f)
    ga.run()