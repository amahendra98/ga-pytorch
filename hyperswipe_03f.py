from ga import GA
import flag_reader

if __name__ == '__main__':
    Trunc = [0.001,0.01,0.05,0.1,0.2,0.3,0.4]
    Pop = [500]

    for t in Trunc:
        for p in Pop:
            flags = flag_reader.read_flag()
            flags.pop_size = p
            flags.trunc_threshold = t
            flags.device = ['cuda:0']
            flags.folder = flags.folder+'/P{}_T{}_value_scheduler'.format(p,t)
            print(flags)
            ga = GA(flags)
            ga.run()

    '''
    Trunc = [0.01]
    Pop = [500, 1000, 2000, 5000]

    count = 0
    for t in Trunc:
        for p in Pop:
            count += 1
            if count <= 0:
                continue

    f = flag_reader.read_flag()
    t = f.trunc_threshold
    f.mutation_power = 0.05
    m = f.mutation_power
    p = f.pop_size
    f.schedule_args = ('generational scheduler', 10, [(0.05,0), (0.02,20),(0.01,120)])

    f.folder = f.folder + "/T{}_M{}_P{}_{}".format(t,m,p,f.schedule_args[0])+'_run5'
    f.device = ['cuda:0']
    print(f)
    ga = GA(f)
    ga.run()
    '''