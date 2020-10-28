from ga import GA
import flag_reader

if __name__ == '__main__':
    Trunc = [0.001,0.01,0.05,0.1,0.2,0.3,0.4]
    Pop = [4000,1000]

    count = 0
    for t in Trunc:
        for p in Pop:
            count += 1
            if count <= 1:
                continue
            flags = flag_reader.read_flag()
            flags.pop_size = p
            flags.trunc_threshold = t
            flags.device = ['cuda:0']
            flags.folder = flags.folder+'/P{}_T{}_value_scheduler'.format(p,t)
            print(flags)
            ga = GA(flags)
            ga.run()

'''
if __name__ == '__main__':
    Thresh = [0.001, 0.0005, 0.0015]
    Small_1 = [5, 10, 10, 10, 10]
    Small_2 = [5, 15, 10, 10, 15]
    Medium = [5, 20, 20, 20, 20]
    Backwards = [5, 40, 30, 25, 15]
    Large_1 = [5, 20, 30, 30, 20]
    Large_2 = [5, 30, 30, 30, 30]
    Increasing = [5, 15, 20, 30, 40]

    Combos = (Small_1, Small_2, Backwards, Medium, Large_1, Large_2, Increasing)
    combo_names = ('sm1', 'sm2', 'bkwd', 'med', 'lrg1', 'lrg2', 'inc')

    count = 0
    for T in Thresh:
        for c, cn in iter(zip(Combos, combo_names)):
            count += 1
            if count <= 2:
                continue

            f = flag_reader.read_flag()
            f.schedule_args = ('variable_length_value_scheduler', 5, [(0.05, 5), (0.02, c[1]),
                                                                      (0.01, c[2]), (0.005, c[3]),
                                                                      (0.001, c[4])], T)
            f.folder = f.folder + '/' + f.schedule_args[0] + '_smallest_' + cn
            f.device = ['cuda:0']
            print(f)
            ga = GA(f)
            ga.run()'''