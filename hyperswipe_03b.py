from ga import GA
import flag_reader

if __name__ == '__main__':
    Pop = [,]
    Trunc = [,,,,,]
    Insertion = [,,,,,]
    K_Nearest = [,,,,,]

    for p in Pop:
        for t in Trunc:
                for i in Insertion:
                    for k in K_Nearest:
                        flags = flag_reader.read_flag()
                        flags.pop_size = p
                        flags.trunc_threshold = t
                        flags.insertion = i
                        flags.k = k
                        flags.folder = "results/two_p_sweep/sweep_03/P{}_T{}_I{}_K{}".format(p,t,i,k)

                        ga = GA(flags)
                        ga.run()