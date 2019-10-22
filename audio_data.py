import numpy as np
import torch
from torch.utils import data

import os

import reduce_samplerate


class AudioDataset(data.Dataset):

    # n_skip will be used to load a test set
    def __init__(self, path, n_per_class=3, n_skip=0):
        super(AudioDataset, self).__init__()
        self.classnames = [n for n in os.listdir(path) if not n.endswith('.mf')]

        songs = []
        labels = []
        for label, classname in enumerate(self.classnames):
            for songname in os.listdir(path + '/' + classname)[n_skip:n_skip+n_per_class]:
                songpath = path + '/' + classname + '/' + songname
                song = reduce_samplerate.load_reduced_wav(songpath)
                songs.append(song)
                labels.append(label)

        self.songs = np.array(songs)
        self.labels = np.array(labels)

    def __getitem__(self, index):
        return self.songs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def oneshot_collate_fn(batch):
    """Takes a list of N samples and returns batch tensors of N/2 pairs.
    N, i.e. batch_size given to DataLoader, needs to be even.

    :param batch: list of samples from AudioDataset.__getitem__
    :return: sample_a, sample_b, same_class: tensors with with first dimension N/2.
        same_class is 1 for slices that are of the same class, otherwise 0
    """
    sample, label = data._utils.collate.default_collate(batch)

    N = label.size(0)
    half_N = int(N / 2)

    p = torch.randperm(N)
    same_class = (label[p[:half_N]] == label[p[-half_N:]]).float()
    return sample[p[:half_N]], sample[p[-half_N:]], same_class


if __name__ == '__main__':
    path = '/home/edvard/SharedWithVirtualBox/genres'
    dataset = AudioDataset(path, n_skip=17)
    print(dataset[-1])
    print(len(dataset))
    dataloader = data.DataLoader(dataset, batch_size=len(dataset), collate_fn=oneshot_collate_fn)
    for _ in range(10):
        batch = next(iter(dataloader))
        print('number of same-class pairs in batch: {}'.format(batch[2].sum()))
