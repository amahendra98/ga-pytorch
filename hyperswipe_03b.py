from ga import GA
import flag_reader

if __name__ == '__main__':
    Pop = [2000,3000]
    Trunc = [0.3,0.4]
    Mutation = [0.01,0.02,0.03]
    K_Nearest = [0.0005,0.001,0.005]

    for p in Pop:
        for t in Trunc:
                for m in Mutation:
                    for k in K_Nearest:
                        flags = flag_reader.read_flag()
                        flags.pop_size = p
                        flags.trunc_threshold = t
                        flags.mutation_power = m
                        flags.k = k
                        flags.device=['cuda:1']
                        flags.folder = "results/two_p_sweep/sweep_03/P{}_T{}_M{}_K{}".format(p,t,m,k)

                        ga = GA(flags)
                        ga.run()