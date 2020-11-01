from ga import GA
import os
import flag_reader

if __name__ == '__main__':
    Layers = [1,2,3,4]
    Nodes = [50,100,200,500]
    Trunc = [0.001, 0.01, 0.1, 0.2,0.3,0.4]
    Pop = [5000,4000,3000,1000]

    for l in Layers:
        for n in Nodes:
            for t in Trunc:
                for p in Pop:
                    for c in range(2):
                        f = flag_reader.read_flag()
                        f.mutation_power = 0.01
                        m = f.mutation_power

                        with open('sweep_07_dirs.txt', 'r') as f:
                            History = eval(f.read())

                        name = 'P{}_T{}_value_scheduler_{}'.format(p,t,c)

                        if name in History:
                            continue

                        History.append(name)

                        f.pop_size = p
                        f.trunc_threshold = t
                        f.linear = [n for j in range(l + 2)]
                        f.linear[0] = 2
                        f.linear[-1] = 300
                        f.schedule_args = [None, None]
                        f.generations = 500
                        f.device = ['cuda:0']
                        f.folder = '/work/amm163/results/sweep_07/'+name

                        print(f)
                        ga = GA(f)
                        ga.run()

    '''
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
                    f.folder = '/work/amm163/results/sweep_04/'+name

                    print(f)
                    ga = GA(f)
                    ga.run()
    '''