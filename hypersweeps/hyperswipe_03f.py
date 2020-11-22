import sys
sys.path.insert(0,sys.path[0]+'\\..')

from ga import GA
import flag_reader

if __name__ == '__main__':
    Nodes = [500]
    Layers = [2,3]

    Params1 = Nodes
    Params2 = Layers

    count = 0
    for p1 in Params1:
        for p2 in Params2:
            count += 1
            for i in range(3):
                if count <= 0:
                    continue
                flags = flag_reader.read_flag()
                flags.pop_size = 3000
                flags.trunc_threshold = 0.01
                flags.generations = 400
                flags.linear = [p1 for i in range(p2 + 2)]
                flags.linear[0] = 2
                flags.linear[-1] = 300
                flags.device = ['cuda:0']
                flags.folder = '/work/amm163/results/2020-11-06/N{}_L{}'.format(p1, p2)
                print(flags)
                ga = GA(flags)
                ga.run()