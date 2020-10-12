import flag_reader
from ga import GA
import torch

if __name__ == '__main__':
    device = ['cuda:0']
    #Trunc = [0.9,0.75,0.5,0.35] #0.65 already taken care of by 20000pop
    #Insertion = [0.001,0.0001] #0.05 taken care of by first pop test #0.01 successful
    #K_Nearest = [0.01,0.005,0.0005,0.0001] #0.001 taken care of by first 500 pop test
    Trunc = [0.3, 0.4]

    for t in Trunc:
        f = flag_reader.read_flag()
        f.pop_size = 3000
        f.trunc_threshold = t
        f.device = device
        f.folder = 'results/two_p_sweep/sweep_02/P{}_T{}_I{}_K{}'.format(f.pop_size,f.trunc_threshold,
                                                                                 f.insertion,f.k)
        print(f)
        ga = GA(f)
        ga.run()


    '''    
        for t in Trunc:
        f = flag_reader.read_flag()
        f.trunc_threshold = t
        f.device = device
        f.folder = 'results/two_p_sweep/sweep_02/P{}_T{}_I{}_K{}'.format(f.pop_size,f.trunc_threshold,
                                                                                 f.insertion,f.k)
        print(f)
        ga = GA(f)
        ga.run()

    for i in Insertion:
        f = flag_reader.read_flag()
        f.insertion = i
        f.device = device
        f.folder = 'results/two_p_sweep/sweep_02/P{}_T{}_I{}_K{}'.format(f.pop_size,f.trunc_threshold,
                                                                                 f.insertion,f.k)
        print(f)
        ga = GA(f)
        ga.run()

    for k in K_Nearest:
        f = flag_reader.read_flag()
        f.k = k
        f.device = device
        f.folder = 'results/two_p_sweep/sweep_02/P{}_T{}_I{}_K{}'.format(f.pop_size,f.trunc_threshold,
                                                                                 f.insertion, f.k)
        print(f)
        ga = GA(f)
        ga.run()

    Insertion = [0.9, 0.8, 0.7, 0.6]
    for i in Insertion:
        f = flag_reader.read_flag()
        f.pop_size = 500
        f.insertion = i
        f.device = device
        f.folder = 'results/two_p_sweep/sweep_02/P{}_T{}_I{}_K{}'.format(f.pop_size, f.trunc_threshold,
                                                                         f.insertion, f.k)
        print(f)
        ga = GA(f)
        ga.run()


    f = flag_reader.read_flag()
    f.pop_size = 1000
    f.insertion = 0.6
    f. device = device
    f.folder = 'results/two_p_sweep/sweep_02/P{}_T{}_I{}_K{}'.format(f.pop_size, f.trunc_threshold,
                                                                     f.insertion, f.k)
    print(f)
    ga = GA(f)
    ga.run()
    '''
