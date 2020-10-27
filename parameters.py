import time

'Run Parameters'
LINEAR = [2,100,100,300]
GENERATIONS = 200
POP_SIZE = 2000
TOP = 0
TRUNC = 0.1
MUT = 0.05
NOV_WEIGHT = 1000
LOSS_WEIGHT = 1
NUM_BATCHES = 1
INSERTION = 0
K_NEAREST = 0.01

'Training Parameters'
SEED = 1234
(y,m,d,hr,min,s,x1,x2,x3) = time.localtime(time.time())
#FOLDER = 'results/two_p/{}{}{}_{}{}{}'.format(y,m,d,hr,min,s)
#FOLDER = '/work/amm163/results/sweep_06'
FOLDER = 'results/sweeps/sweep_06'
DEVICE_LIST = ['cuda:0']
SCHED_ARGS = ('variable length value scheduler', 5, [(MUT,5), (0.02,20),(0.01,)])

'Gif Parameters'
TIME = 10
XLIM = (-2, 2)
YLIM = (-2, 2)


