import oneshot_audio_data

import torch
from torch import nn, optim
from torch.utils import data

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/edvard/SharedWithVirtualBox/genres')
parser.add_argument('--epochs', type=int, default=100)


class CNN1D(nn.Module):
    # TODO idea: might need some form of norm to be able to train well
    # TODO idea: use dilation in initial convs, I am thinking about the 5000 kernel size in block1 especially
    # TODO idea: initialize weights with random walk

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


class SiameseNet(nn.Module):

    def __init__(self):
        super(SiameseNet, self).__init__()
        self.cnn1d = CNN1D()
        self.linear = nn.Linear(80, 1)

    def forward(self, sample_a, sample_b):
        embedding_a = torch.selu(self.cnn1d(sample_a))
        embedding_b = torch.selu(self.cnn1d(sample_b))
        abs_difference = torch.abs(embedding_a - embedding_b)
        logit_score = self.linear(abs_difference).squeeze(-1)
        return logit_score


if __name__ == '__main__':
    args = parser.parse_args()

    dataset = oneshot_audio_data.AudioDataset(args.path)
    dataloader = data.DataLoader(dataset, batch_size=len(dataset), collate_fn=oneshot_audio_data.oneshot_collate_fn)

    net = SiameseNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(args.epochs):
        sample_a, sample_b, same_class = next(iter(dataloader))
        optimizer.zero_grad()
        output = net(sample_a, sample_b)
        loss = criterion(output, same_class)
        loss.backward()
        optimizer.step()

        print('epochs of training: {}, loss of current epoch: {}'.format(epoch, loss))
