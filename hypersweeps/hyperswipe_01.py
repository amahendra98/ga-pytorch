import sys
sys.path.append(sys.path[0]+'\\..')
print(sys.path)
import flag_reader
from ga import GA

if __name__ == '__main__':
    Pop = [50, 100, 200, 500]
    Trunc = [0.02, 0.05,0.1,0.15,0.2]
    Mut = [0.02,0.01,0.001,0.1,0.5,1]
    Insertion = [0.01,0.05,0.1,0.2,0.5]
    K_Nearest = Trunc

    count = 0

    for p in Pop:
        for t in Trunc:
            for m in Mut:
                for i in Insertion:
                    for k in K_Nearest:
                        if count<126:
                            count += 1
                            continue

                        flags = flag_reader.read_flag()
                        flags.pop_size = p
                        flags.trunc_threshold = t
                        flags.mutation_power = m
                        flags.insertion = i
                        flags.k = k

                        flags.generations = 500
                        flags.top = 0
                        flags.folder = "results/sweeps/sweep_01/P{}_T{}_M{}_I{}_K{}".format(p,t,m,i,k)

                        ga = GA(flags)
                        ga.run()