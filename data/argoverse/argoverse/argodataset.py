import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from skimage.transform import rotate



class ArgoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        data = self.data[0][idx]
        label = self.data[1][idx]
        return data, label


    def __len__(self):
        return len(self.data[0])
