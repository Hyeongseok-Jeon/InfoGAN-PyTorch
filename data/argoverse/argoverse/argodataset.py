
from torch.utils.data import Dataset


class ArgoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        data = self.data[0][idx]
        label = self.data[1][idx]
        return data, label


    def __len__(self):
        return len(self.data[0])
