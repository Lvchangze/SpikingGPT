import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from torch.utils.data import Dataset

import numpy as np





class CharDataset(Dataset):

    def __init__(self, data_path, block_size):
        data = open(data_path, 'r').read()
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y














