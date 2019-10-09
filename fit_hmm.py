import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='/home/edvard/SharedWithVirtualBox/corn2015-2017/corn2013-2017.txt')
parser.add_argument('--data-subset', default='/home/edvard/SharedWithVirtualBox/corn2015-2017/corn2015-2017.txt')
parser.add_argument('--some-third-file', default='/home/edvard/SharedWithVirtualBox/corn2015-2017/corn_OHLC2013-2017.txt')


if __name__ == '__main__':
    args = parser.parse_args()

    data_frame = pd.read_csv(args.dataset)
    subset_frame = pd.read_csv(args.data_subset)

    N = 10
    model = hmm.GaussianHMM(n_components=N)
    #model.startprob_ = np.array([1 / N] * N)
    #model.transmat_ = np.random.rand(N, N)  # needs to be "ergodic" (idk what this means)
    #print(data_frame.iloc[:,1])
    model.fit(data_frame.iloc[:,1].to_numpy().reshape(-1, 1))
    print(model.startprob_)
    print(model.transmat_)
    print(model.transmat_prior)

    sample_x, sample_z = model.sample(100)

