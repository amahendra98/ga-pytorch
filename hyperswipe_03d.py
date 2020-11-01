from ga import GA
import os
import flag_reader

if __name__ == '__main__':
    Trunc = [0.001, 0.01, 0.1, 0.2,0.3,0.4]
    Pop = [3000,1000]

    for t in Trunc:
        for p in Pop:
            for c in range(2):
                f = flag_reader.read_flag()
                f.mutation_power = 0.01
                m = f.mutation_power

                        #with open('sweep_07_dirs.txt', 'r') as f:
                        #    History = eval(f.read())

                name = 'P{}_T{}_value_scheduler_{}'.format(p,t,c)

                '''
                if name in History:
                    continue

                History.append(name)
                with open('sweep_07_dirs.txt', 'w') as f:
                    f.truncate()
                    print(History, file=f)
                '''

                f.pop_size = p
                f.trunc_threshold = t
                f.generations = 200
                f.device = ['cuda:0']
                f.folder = '/work/amm163/results/sweep_07/'+name

                print(f)
                ga = GA(f)
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