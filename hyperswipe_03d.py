from ga import GA
import flag_reader

if __name__ == '__main__':
    Layers = [1,2,3,4]
    Nodes = [50,100,200,500]
    Trunc = [0.001, 0.01, 0.1, 0.2,0.3,0.4]
    Pop = [5000,4000,3000,1000]

    History = None
    with open('sweep_04_dirs.txt','r') as f:
        History = eval(f.read())

    for l in Layers:
        for n in Nodes:
            for t in Trunc:
                for p in Pop:
                    f = flag_reader.read_flag()
                    f.mutation_power = 0.01
                    m = f.mutation_power
                    name = 'T{}_M{}_P{}_L{}_N{}'.format(t,m,p,l,n)

                    if name in History:
                        continue

                    f.pop_size = p
                    f.trunc_threshold = t
                    f.linear = [n for j in range(l + 2)]
                    f.linear[0] = 2
                    f.linear[-1] = 300
                    f.schedule_args = [None,None]
                    f.generations = 500
                    f.device = ['cuda:0']
                    f.folder = '/work/amm163/results/sweep_04'+name

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