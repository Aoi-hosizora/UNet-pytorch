import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import FFDNet

def load_dataset(dataset_path):
    """
    Load train/test datasets from dataset_path
    """
    pass

def train():
    """
    Train UNet from datasets
    """
    pass


def test():
    """
    Test some data from trained UNet
    """
    pass


def main():
    load_dataset('./dataset/')
    pass


if __name__ == '__main__':
    main()
