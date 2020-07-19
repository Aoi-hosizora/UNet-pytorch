import argparse
import time
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import UNet, loss_fn
from dataset import SSDataset, load_test_image


def train(args):
    """
    Train UNet from datasets
    """

    # dataset
    print('Reading dataset from {}...'.format(args.dataset_path))
    train_dataset = SSDataset(dataset_path=args.dataset_path, is_train=True)
    val_dataset = SSDataset(dataset_path=args.dataset_path, is_train=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    # mask
    with open(args.mask_json_path, 'w', encoding='utf-8') as mask:
        colors = SSDataset.all_colors
        mask.write(json.dumps(colors))
        print('Mask colors list has been saved in {}'.format(args.mask_json_path))

    # model
    net = UNet(in_channels=3, out_channels=5)
    if args.cuda:
        net = net.cuda()

    # setting
    lr = args.lr  # 1e-3
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = loss_fn

    # run
    train_losses = []
    val_losses = []
    print('Start training...')
    for epoch_idx in range(args.epochs):
        # train
        net.train()
        train_loss = 0
        for batch_idx, batch_data in enumerate(train_dataloader):
            xs, ys = batch_data
            if args.cuda:
                xs = xs.cuda()
                ys = ys.cuda()
            ys_pred = net(xs)
            loss = criterion(ys_pred, ys)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # val
        net.eval()
        val_loss = 0
        for batch_idx, batch_data in enumerate(val_dataloader):
            xs, ys = batch_data
            if args.cuda:
                xs = xs.cuda()
                ys = ys.cuda()
            ys_pred = net(xs)
            loss = loss_fn(ys_pred, ys)
            val_loss += loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print('Epoch: {}, Train total loss: {}, Val total loss: {}'.format(epoch_idx + 1, train_loss.item(), val_loss.item()))

        # save
        if (epoch_idx + 1) % args.save_epoch == 0:
            checkpoint_path = os.path.join(args.checkpoint_path, 'checkpoint_{}.pth'.format(epoch_idx + 1))
            torch.save(net.state_dict(), checkpoint_path)
            print('Saved Checkpoint at Epoch {} to {}'.format(epoch_idx + 1, checkpoint_path))

    # summary
    if args.do_save_summary:
        epoch_range = list(range(1, args.epochs + 1))
        plt.plot(epoch_range, train_losses, 'r', label='Train loss')
        plt.plot(epoch_range, val_loss, 'g', label='Val loss')
        plt.imsave(args.summary_image)
        print('Summary images have been saved in {}'.format(args.summary_image))

    # save
    net.eval()
    torch.save(net.state_dict(), args.model_state_dict)
    print('Saved state_dict in {}'.format(args.model_state_dict))


def test(args):
    """
    Test some data from trained UNet
    """
    image = load_test_image(args.test_image) # 1 c w h
    net = UNet(in_channels=3, out_channels=5)
    if args.cuda:
        net = net.cuda()
        image = image.cuda()
    print('Loading model param from {}'.format(args.model_state_dict))
    net.load_state_dict(torch.load(args.model_state_dict))
    net.eval()

    print('Predicting for {}...'.format(args.test_image))
    ys_pred = net(image)  # 1 ch w h

    colors = []
    with open(args.mask_json_path, 'r', encoding='utf-8') as mask:
        print('Reading mask colors list from {}'.format(args.mask_json_path))
        colors = json.loads(mask.read())
        colors = [tuple(c) for c in colors]
        print('Mask colors: {}'.format(colors))

    ys_pred = ys_pred.cpu().detach().numpy()[0]
    ys_pred[ys_pred < 0.5] = 0
    ys_pred[ys_pred >= 0.5] = 1
    ys_pred = ys_pred.astype(np.int)
    image_w = ys_pred.shape[1]
    image_h = ys_pred.shape[2]
    out_image = np.zeros((image_w, image_h, 3))

    for w in range(image_w):
        for h in range(image_h):
            for ch in range(ys_pred.shape[0]):
                if ys_pred[ch][w][h] == 1:
                    out_image[w][h][0] = colors[ch][0]
                    out_image[w][h][1] = colors[ch][1]
                    out_image[w][h][2] = colors[ch][2]

    out_image = out_image.astype(np.uint8) # w h c
    out_image = out_image.transpose((1, 0, 2)) # h w c
    out_image = Image.fromarray(out_image)
    out_image.save(args.test_save_path)
    print('Segmentation result has been saved to {}'.format(args.test_save_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='Do train')
    parser.add_argument('--do_test', action='store_true', help='Do test')
    parser.add_argument('--no_gpu', action='store_true', help='Do not use GPU to train and test')
    parser.add_argument('--dataset_path', type=str, default='./dataset/', help='Dataset folder path')
    parser.add_argument('--mask_json_path', type=str, default='./mask.json', help='Mask json path')
    parser.add_argument('--batch_size', type=int, default=128, help='Train and validate batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Train epoch number')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--save_epoch', type=int, default=1, help='Save checkpoint every epoch')
    parser.add_argument('--checkpoint_path', type=str, default='./model/', help='Model checkpoint save path')
    parser.add_argument('--model_state_dict', type=str, default='./model/model.pth', help='Model load and sav path')
    parser.add_argument('--do_save_summary', action='store_true', help='Do save summary image')
    parser.add_argument('--summary_image', type=str, default='./summary.png', help='Summary image save path')
    parser.add_argument('--test_image', type=str, default='./test.png', help='Test image path')
    parser.add_argument('--test_save_path', type=str, default='./test_seg.png', help='Test predict save path')
    args = parser.parse_args()
    assert (args.do_train or args.do_test), 'You must do train or test'

    args.cuda = not args.no_gpu and torch.cuda.is_available()
    print('\nParameters: ')
    for k, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print('\t{}: {}'.format(k, v))
    print('\n')

    if args.do_train:
        train(args)

    if args.do_test:
        test(args)


if __name__ == '__main__':
    main()
