
import numpy as np
from .utils import load_cifar10, standardize_per_channel

class DataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True):
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle
        self.indices = np.arange(len(X))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start in range(0, len(self.X), self.batch_size):
            idx = self.indices[start:start+self.batch_size]
            yield self.X[idx], self.y[idx]

def load_cifar10_loaders(path="data", batch_size=64, standardize=True):
    Xtr, ytr = load_cifar10(path, "train")
    Xte, yte = load_cifar10(path, "test")
    if standardize:
        Xtr, mean, std = standardize_per_channel(Xtr)
        Xte, _, _ = standardize_per_channel(Xte, mean, std)
    return DataLoader(Xtr, ytr, batch_size=batch_size, shuffle=True), DataLoader(Xte, yte, batch_size=batch_size, shuffle=False)
