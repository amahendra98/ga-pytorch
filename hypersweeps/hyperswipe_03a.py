import sys
sys.path.insert(0,sys.path[0]+'\\..')

from ga import GA
import flag_reader

if __name__ == '__main__':
    Trunc = [0.01]
    Pop = [10]

    count = 0
    for t in Trunc:
        for p in Pop:
            count += 1
            for i in range(6,9):
                if count <= 0:
                    continue
                flags = flag_reader.read_flag()
                flags.pop_size = p
                flags.trunc_threshold = t
                flags.generations = 3000
                flags.device = ['cuda:1']
                flags.folder = 'results/2020-11-05/P{}_T{}_value_scheduler_end{}'.format(p,t,i)
                print(flags)
                ga = GA(flags)
                ga.run()

    '''
    Thresh = [0.001,0.0005, 0.0015]
    Small_1 = [5,5,5,5,5,5,5,5]
    Small_2 = [5,5,5,5,10,10,10,10]
    Medium = [5,10,10,10,10,10,10,10]
    Backwards = [5,20,15,10,10,5,5,5]
    Large_1 = [5,15,15,15,15,15,15,15]
    Large_2 = [5,20,20,20,20,20,20,20]
    Increasing = [5,5,5,10,10,15,20,30]

    Combos = (Small_1, Small_2, Backwards, Medium, Large_1, Large_2, Increasing)
    combo_names = ('sm1', 'sm2','bkwd','med','lrg1','lrg2','inc')

    count = 0
    for T in Thresh:
        for c,cn in iter(zip(Combos, combo_names)):
            count += 1
            if count <= 1:
                continue

            f = flag_reader.read_flag()
            f.schedule_args = ('variable_length_value_scheduler', 5, [(0.05, 5), (0.02, c[1]), (0.015,c[2]),
                                                                      (0.01, c[3]), (0.0075, c[4]), (0.005, c[5]),
                                                                      (0.0025, c[6]), (0.001, c[7])], T)
            f.folder = f.folder+'/'+f.schedule_args[0]+'_longest_'+cn
            f.device = ['cuda:1']
            print(f)
            ga = GA(f)
            ga.run()
    
    '''