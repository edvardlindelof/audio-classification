import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='/home/edvard/SharedWithVirtualBox/corn2015-2017/corn2013-2017.txt')
parser.add_argument('--data-subset', default='/home/edvard/SharedWithVirtualBox/corn2015-2017/corn2015-2017.txt')
parser.add_argument('--some-third-file', default='/home/edvard/SharedWithVirtualBox/corn2015-2017/corn_OHLC2013-2017.txt')


if __name__ == '__main__':
    args = parser.parse_args()

    print('third file description:')
    print(pd.read_csv(args.some_third_file).describe())

    data_frame = pd.read_csv(args.dataset)
    subset_frame = pd.read_csv(args.data_subset)

    fig, axs = plt.subplots(1, 2)
    data_frame.plot(ax=axs[0])
    subset_frame.plot(ax=axs[1])
    plt.show()
