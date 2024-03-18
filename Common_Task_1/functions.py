import torch
from torch.utils.data import Dataset
import h5py

class H5Dataset(Dataset):
    def __init__(self, filename):
        self.f = h5py.File(filename, 'r')
        print('The following keys are availible: ', list(self.f.keys()))
    def load_data(self,inputs,targets):
        self.X = self.f[inputs]
        print('Loaded inputs tensor ', self.X.shape)
        self.y = self.f[targets]
        print('Loaded targets tensor ', self.y.shape)
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

def split_Dataset(full_dataset, split_ratio=0.8):
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    return torch.utils.data.random_split(full_dataset, [train_size, test_size])