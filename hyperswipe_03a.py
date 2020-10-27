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
    f.schedule_args = ('variable_length_value_scheduler', 5, [(0.05,5), (0.02, 10),(0.01,20),(0.005, 30)], 0.001)

    f.folder = f.folder + "/T{}_M{}_P{}_{}".format(t,m,p,f.schedule_args[0])+'_run0'
    f.device = ['cuda:1']
    print(f)
    ga = GA(f)
    ga.run()