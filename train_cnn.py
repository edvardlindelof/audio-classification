import audio_data

import torch
from torch import nn, optim
from torch.utils import data
import numpy as np

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/edvard/SharedWithVirtualBox/genres')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--validate-every', type=int, default=10)

parser.add_argument('--light', action='store_true', help='Lighter setting for debugging etc')


class CNN1D(nn.Module):

    def __init__(self):
        super(CNN1D, self).__init__()
        self.block1 = nn.Sequential(nn.Conv1d(1, 5, 5000, 100), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv1d(5, 25, 30, 10), nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv1d(25, 125, 10, 10), nn.ReLU())
        #self.block4 = nn.Sequential(nn.Conv1d(125, 625, 10, 10), nn.ReLU())
        self.linear = nn.Linear(125, 80)

    def forward(self, input):
        x0 = input.unsqueeze(1)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        #x4 = self.block4(x3)
        embedding = self.linear(x3.squeeze(-1))
        return embedding


class ClassificationCNN1D(nn.Module):

    def __init__(self):
        super(ClassificationCNN1D, self).__init__()
        self.cnn = CNN1D()
        self.linear = nn.Linear(80, 10)

    def forward(self, input):
        embedding = torch.selu(self.cnn(input))
        pred_logits = self.linear(embedding)
        return pred_logits


if __name__ == '__main__':
    args = parser.parse_args()

    if args.light:
        samples_per_class_train = 3
        samples_per_class_val = 3
    else:
        samples_per_class_train = 80
        samples_per_class_val = 20

    dataset_train = audio_data.AudioDataset(args.path, n_per_class=samples_per_class_train)
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataset_val = audio_data.AudioDataset(args.path, n_per_class=samples_per_class_val, n_skip=80)
    dataloader_val = data.DataLoader(dataset_val, batch_size=100)

    net = ClassificationCNN1D()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(args.epochs):
        song, label = next(iter(dataloader_train))
        optimizer.zero_grad()
        output = net(song)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if epoch % args.validate_every == 0:
            with torch.no_grad():
                song, label = next(iter(dataloader_val))
                output = net(song)
                label_pred = np.argmax(output, axis=1)
                accuracy = np.mean(np.array(label_pred) == np.array(label))

                print('epochs of training: {}, loss at current epoch: {}'.format(epoch, loss))
                print('validation accuracy: {}'.format(accuracy))
