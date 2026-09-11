import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None, indices=None):
        self.X = X[indices] if indices is not None else X
        self.Y = Y[indices] if indices is not None else Y
        self.transform = transform

        self.indices = indices

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.open(x).convert("RGB")
            x = self.transform(x)
        return {'x': x, 'y': y, 'index': index}

    def __len__(self):
        return len(self.X)
