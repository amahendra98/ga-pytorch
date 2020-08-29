import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Define the class for the Meta-material dataset
class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]

## Copied from Omar's code

# Make geometry samples
def MM_Geom(n):
    # Parameter bounds for metamaterial radius and height
    r_min = 20
    r_max = 200
    h_min = 20
    h_max = 100

    # Defines hypergeometric space of parameters to choose from
    space = 10
    r_space = np.linspace(r_min, r_max, space + 1)
    h_space = np.linspace(h_min, h_max, space + 1)

    # Shuffles r,h arrays each iteration and then selects 0th element to generate random n x n parameter set
    r, h = np.zeros(n, dtype=float), np.zeros(n, dtype=float)
    for i in range(n):
        np.random.shuffle(r_space)
        np.random.shuffle(h_space)
        r[i] = r_space[0]
        h[i] = h_space[0]
    return r, h

# Make geometry and spectra
def Make_MM_Model(n):

    r, h = MM_Geom(n)
    spectra = np.zeros(300)
    geom = np.concatenate((r, h), axis=0)
    for i in range(n):
        w0 = 100 / h[i]
        wp = (1 / 100) * np.sqrt(np.pi) * r[i]
        g = (1 / 1000) * np.sqrt(np.pi) * r[i]
        w, e2 = Lorentzian(w0, wp, g)
        spectra += e2
    return geom, spectra

# Calculate Lorentzian function to get spectra
def Lorentzian(w0, wp, g):

    freq_low = 0
    freq_high = 5
    num_freq = 300
    w = np.arange(freq_low, freq_high, (freq_high - freq_low) / num_freq)

    # e1 = np.divide(np.multiply(np.power(wp, 2), np.add(np.power(w0, 2), -np.power(w, 2))),
    #                   np.add(np.power(np.add(np.power(w0, 2), -np.power(w, 2)), 2),
    #                          np.multiply(np.power(w, 2), np.power(g, 2))))

    e2 = np.divide(np.multiply(np.power(wp, 2), np.multiply(w, g)),
                   np.add(np.power(np.add(np.power(w0, 2), -np.power(w, 2)), 2),
                          np.multiply(np.power(w, 2), np.power(g, 2))))
    return w, e2

# Generates randomized dataset of simulated spectra for training and testing
def Prepare_Data(osc, sets, batch_size):

    features = []
    labels = []

    for i in range(sets):
        geom, spectra = Make_MM_Model(osc)
        features.append(geom)
        labels.append(spectra)

    features = np.array(features, dtype='float32')
    labels = np.array(labels, dtype='float32')

    ftrsize = features.size / sets
    lblsize = labels.size / sets
    print('Size of Features is %i, Size of Labels is %i' % (ftrsize, lblsize))
    print('There are %i datasets:' % sets)

    ftrTrain, ftrTest, lblTrain, lblTest = train_test_split(features, labels, test_size=0.2, random_state=1234)
    train_data = MetaMaterialDataSet(ftrTrain, lblTrain, bool_train=True)
    test_data = MetaMaterialDataSet(ftrTest, lblTest, bool_train=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    print('Number of Training samples is {}'.format(len(ftrTrain)))
    print('Number of Test samples is {}'.format(len(ftrTest)))
    return train_loader, test_loader

def gen_data(name):
    train_loader, test_loader = Prepare_Data(1, 10000, 1000)

    with open(name, 'a') as datafile:
        for j, (geometry, spectra) in enumerate(train_loader):
            concate = np.concatenate([geometry, spectra], axis=1)
            # print(np.shape(concate))
            np.savetxt(datafile, concate, delimiter=',')


if __name__ == "__main__":
    train_loader, test_loader = Prepare_Data(1, 10000, 1000)

    with open('toy_data/mm1d_6.csv', 'a') as datafile:
        for j, (geometry, spectra) in enumerate(train_loader):
            concate = np.concatenate([geometry, spectra], axis=1)
            #print(np.shape(concate))
            np.savetxt(datafile, concate, delimiter=',')