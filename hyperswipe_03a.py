from ga import GA
import flag_reader

if __name__ == '__main__':
    Trunc = [0.001,0.01,0.05,0.1,0.2,0.3,0.4]
    Pop = [5000]

    for t in Trunc:
        for p in Pop:
            flags = flag_reader.read_flag()
            flags.pop_size = p
            flags.trunc_threshold = t
            flags.device = ['cuda:1']
            flags.folder = flags.folder+'/P{}_T{}_value_scheduler'
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