import time
import torch
from inspect import currentframe

# def get_linenumber():
#     cf = currentframe()
#     return cf.f_back.f_lineno

class Pole(object):
    def __init__(self,name):
        super(Pole).__init__()
        self.name = name
        self.time = time.time()

class TimeClock(object):
    def __init__(self):
        super(TimeClock).__init__()
        self.Poles = []
        self.Pairs = []
        self.Sets = [0,0,0,0,0]

    def drop_pole(self, name):
        self.Poles.append(Pole(name))

    def report(self):
        print("TIME REPORT:\n\n")
        start_time = 0
        prev_name = None
        report_string = ""
        print("Sequential Analysis")
        for i in range(len(self.Poles) - 1):
            set = i%5
            p1 = self.Poles[i]
            p2 = self.Poles[i+1]
            elapsed = p2.time - p1.time
            self.Sets[set] += elapsed
        for i in range(5):
            p1 = self.Poles[i]
            p2 = self.Poles[i + 1]
            print('{} - {}:\t{}'.format(p1.name,p2.name,self.Sets[i]))

        print("\nPairs of Interest")
