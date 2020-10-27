from ga import GA
import flag_reader

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

    for T in Thresh:
        for c, cn in iter(zip(Combos, combo_names)):
            f = flag_reader.read_flag()
            f.schedule_args = ('variable_length_value_scheduler', 5, [(0.05, 5), (0.02, c[1]),
                                                                      (0.01, c[2]), (0.005, c[3]),
                                                                      (0.001, c[4])], T)
            f.folder = f.folder + '/' + f.schedule_args[0] + '_longest_' + cn
            f.device = ['cuda:0']
            print(f)
            ga = GA(f)
            ga.run()