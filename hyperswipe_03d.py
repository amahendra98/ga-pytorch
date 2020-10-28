from ga import GA
import flag_reader

if __name__ == '__main__':

    Trunc = [0.001,0.01,0.05,0.1,0.2,0.3,0.4]
    Pop = [2000]

    for t in Trunc:
        for p in Pop:
            flags = flag_reader.read_flag()
            flags.pop_size = p
            flags.trunc_threshold = t
            flags.device = ['cuda:0']
            flags.folder = flags.folder+'/P{}_T{}_value_scheduler'
            print(flags)
            ga = GA(flags)
            ga.run()

    '''Small_1 = [5,5,5,5,5,5]
    Small_2 = [5,5,5,10,10,10]
    Medium = [5,15,15,15,15,15]
    Backwards = [5,40,30,25,15,10]
    Large_1 = [5,20,20,20,20,20]
    Large_2 = [5,30,20,30,20,30]
    Increasing = [5,10,15,20,30,40]

    Combos = (Small_1, Small_2, Backwards, Medium, Large_1, Large_2, Increasing)
    combo_names = ('sm1', 'sm2','bkwd','med','lrg1','lrg2','inc')

    for c,cn in iter(zip(Combos, combo_names)):
        f = flag_reader.read_flag()
        f.schedule_args = ('variable_length_value_scheduler', 5, [(0.05, 5), (0.02, c[1]), (0.015,c[2]),
                                                                  (0.01, c[3]), (0.005, c[4]),
                                                                  (0.001, c[5])], 0.001)
        f.folder = f.folder+'/'+f.schedule_args[0]+'_smaller_'+cn
        f.device = ['cuda:1']
        print(f)
        ga = GA(f)
        ga.run()
    '''