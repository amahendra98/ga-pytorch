from ga import GA
import flag_reader

if __name__ == '__main__':
    Trunc = [0.01]
    Mutation = [0.01, 0.1, 1]
    Pop = [500, 1000, 2000, 5000]
    Layers = [1, 2, 3, 4]
    Nodes = [50,100,200,500]

    count = 0
    for t in Trunc:
        for m in Mutation:
            for p in Pop:
                for l in Layers:
                    for n in Nodes:
                        count += 1
                        if count >= 2:
                            continue
                        if l == 1 or n == 100 or l*n*p >= 3*1000000:
                            continue

                        f = flag_reader.read_flag()
                        t = f.trunc_threshold
                        f.mutation_power = 0.5
                        p = f.pop_size

                        f.folder = f.folder + "/T{}_M{}_P{}".format(t, m, p)
                        f.device = ['cuda:0']
                        print(f)
                        ga = GA(f)
                        ga.run()

                        """flags = flag_reader.read_flag()
                        flags.pop_size = p
                        flags.trunc_threshold = t
                        flags.mutation_power = m
                        flags.linear = [2] + [n for z in range(l) ] +[300]
                        flags.device = ['cuda:0']
                        flags.folder = flags.folder+"/T{}_M{}_P{}_L{}_N{}".format(t,m,p,l,n)

                        #ga = GA(flags)
                        #ga.run()"""