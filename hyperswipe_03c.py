from ga import GA
import flag_reader

if __name__ == '__main__':
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
    f.schedule_args = ('generational scheduler', 10, [(0.05,0), (0.02,20),(0.01,100)])

    f.folder = f.folder + "/T{}_M{}_P{}_{}".format(t,m,p,f.schedule_args[0])+'_run2'
    f.device = ['cuda:0']
    print(f)
    ga = GA(f)
    ga.run()