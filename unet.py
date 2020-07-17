import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import UNet, loss_fn
from dataset import SSDataset


def train():
    """
    Train UNet from datasets
    """
    # hyperparameter
    batch_size = 128
    epochs = 5

    # dataset
    train_dataset = SSDataset(dataset_path='./dataset/', is_train=True)
    test_dataset = SSDataset(dataset_path='./dataset/', is_train=False)
    train_dataloader = DataLoader(dataset=terain_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # model
    net = UNet(in_channels=3, out_channels=4)
    if torch.cuda.is_available:
        net = net.cuda()

    # setting
    lr = 1e-3
    momentum = 0.99
    optimizer = optim.Adam(net.parameters(), lr=lr, momentum=momentum)
    loss_fn = loss_fn

    # train
    for epoch_idx in range(epochs):
        model.train()
        for batch_idx, batch_data in enumerate(train_dataloader):
            xs, ys = batch_data
            if torch.cuda.is_available:
                xs = xs.cuda()
                ys = ys.cuda()
            ys_pred = net(xs)

            loss = loss_fn(ys_pred, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        losses = 0
        for batch_idx, batch_data in enumerate(test_dataloader):
            xs, ys = batch_data
            if torch.cuda.is_available:
                xs = xs.cuda()
                ys = ys.cuda()
            ys_pred = net(xs)
            losses += loss_fn(ys_pred, ys)

        print('Epoch: {}, Loss: {}', epoch_idx, losses)


def test():
    """
    Test some data from trained UNet
    """
    pass


def main():
    train()
    pass


if __name__ == '__main__':
    main()
