from ga import GA
import flag_reader

if __name__ == '__main__':
    Pop = [5000]
    Trunc = [0.4]
    Mutation = [0.02]
    K_Nearest = [0.0005,0.001,0.005]

    count = 0
    for p in Pop:
        for t in Trunc:
                for m in Mutation:
                    for k in K_Nearest:
                        count += 1
                        if count <= 0:
                            continue
                        flags = flag_reader.read_flag()
                        flags.pop_size = p
                        flags.trunc_threshold = t
                        flags.mutation_power = m
                        flags.k = k
                        flags.device = ['cuda:0']
                        flags.folder = flags.folder+"/P{}_T{}_M{}_K{}".format(p,t,m,k)

                        ga = GA(flags)
                        ga.run()