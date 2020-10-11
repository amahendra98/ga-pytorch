import time

'Run Parameters'
GENERATIONS = 100
POP_SIZE = 2000
TOP = 0
TRUNC = 0.65
MUT = 0.02
NOV_WEIGHT = 1000
LOSS_WEIGHT = 0
NUM_BATCHES = 1
INSERTION = 0.001
K_NEAREST = 0.001

'Training Parameters'
SEED = 1234
(y,m,d,hr,min,s,x1,x2,x3) = time.localtime(time.time())
FOLDER = 'results/two_p/{}{}{}_{}{}{}'.format(y,m,d,hr,min,s)
DEVICE_LIST = ['cuda:1']

'Gif Parameters'
TIME = 10
XLIM = (-2, 2)
YLIM = (-2, 2)


