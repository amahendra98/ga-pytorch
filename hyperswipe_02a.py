import flag_reader
from ga import GA
import torch

if __name__ == '__main__':
    device = ['cuda:1']
    Pop = [3000,6000]

    for p in Pop:
        f = flag_reader.read_flag()
        f.pop_size = p
        f.trunc_threshold = 0.5
        f.device = device
        f.folder = 'results/two_p_sweep/sweep_02/P{}_T{}_I{}_K{}'.format(f.pop_size,f.trunc_threshold,
                                                                                 f.insertion,f.k)
        print(f)
        ga = GA(f)
        ga.run()