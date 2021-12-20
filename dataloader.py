import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import glob
import csv
import pandas as pd
import os
import numpy as np
import data.argoverse.argoverse.cord_conv as cord_conv
import matplotlib.pyplot as plt
import time
# Directory containing the data.
root = 'data/'

def get_data(dataset, batch_size):
    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root+'mnist/', train='train', 
                                download=True, transform=transform)

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = dsets.SVHN(root+'svhn/', split='train', 
                                download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root+'fashionmnist/', train='train', 
                                download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root+'celeba/', transform=transform)

    elif dataset == 'argoverse':
        traj_num = 0
        data_placeholder = np.empty(shape=(2000000, 20, 2))
        maneuver = np.empty(shape=2000000)
        os.makedirs(root+'argoverse/argoverse/preprocess', exist_ok=True)
        data_list = glob.glob(root+'argoverse/argoverse/raw/*.csv')
        data_num = len(data_list)
        for i in range(data_num):
            ID_cand = []
            data_tmp = pd.read_csv(data_list[i])
            ts = list(data_tmp.TIMESTAMP[~data_tmp.TIMESTAMP.duplicated()])[20:40]
            ID_cand.append(list(data_tmp.TRACK_ID[data_tmp.OBJECT_TYPE == 'AGENT'])[0])
            ID_cand.append(list(data_tmp.TRACK_ID[data_tmp.OBJECT_TYPE == 'AV'])[0])
            for j in range(len(ID_cand)):
                ID = ID_cand[j]
                if len(set(ts) & set(list(data_tmp.TIMESTAMP[data_tmp.TRACK_ID == ID]))) == 20:
                    X = list(data_tmp.X[data_tmp.TRACK_ID == ID])
                    Y = list(data_tmp.Y[data_tmp.TRACK_ID == ID])
                    if np.sqrt((X[-1]-X[0])**2 + (Y[-1]-Y[0])**2) > 5:
                        t = list(data_tmp.TIMESTAMP[data_tmp.TRACK_ID == ID])
                        traj_idx = [i for i in range(len(t)) if t[i] in ts]
                        traj_tmp = np.expand_dims(np.asarray([X[traj_idx[0]:traj_idx[-1]+1], Y[traj_idx[0]:traj_idx[-1]+1]]).T, axis=0)
                        traj_tmp = cord_conv.preprocess(traj_tmp)
                        traj_tmp = cord_conv.preprocess_dir(traj_tmp)[0]

                        lat_pos_final = traj_tmp[-1, 1]
                        heading_final = np.rad2deg(np.arctan2(traj_tmp[-1, 1]-traj_tmp[-5, 1], traj_tmp[-1, 0]-traj_tmp[-5, 0]))

                        if heading_final > 10:
                            maneuver_cat = 0
                            plt.subplot(5,1,1)
                            plt.plot(traj_tmp[:,0], traj_tmp[:,1])
                            plt.xlim([0, 50])
                            plt.ylim([-5, 5])
                        elif heading_final > -10:
                            if lat_pos_final > 1:
                                plt.subplot(5, 1, 2)
                                plt.plot(traj_tmp[:, 0], traj_tmp[:, 1])
                                plt.xlim([0, 50])
                                plt.ylim([-5, 5])
                                maneuver_cat = 1
                            elif lat_pos_final > -1:
                                plt.subplot(5, 1, 3)
                                plt.plot(traj_tmp[:, 0], traj_tmp[:, 1])
                                plt.xlim([0, 50])
                                plt.ylim([-5, 5])
                                maneuver_cat = 2
                            else:
                                plt.subplot(5, 1, 4)
                                plt.plot(traj_tmp[:, 0], traj_tmp[:, 1])
                                plt.xlim([0, 50])
                                plt.ylim([-5, 5])
                                maneuver_cat = 3
                        else:
                            plt.subplot(5,1,5)
                            plt.plot(traj_tmp[:,0], traj_tmp[:,1])
                            plt.xlim([0, 50])
                            plt.ylim([-5, 5])
                            maneuver_cat = 4

                        data_placeholder[traj_num] = traj_tmp
                        maneuver[traj_num] = maneuver_cat
                        traj_num = traj_num + 1

        data_placeholder = torch.Tensor(data_placeholder[:traj_num])
        maneuver = torch.Tensor(maneuver[:traj_num])
        training_data = (data_placeholder, maneuver)
        torch.save(root+'argoverse/argoverse/preprocess/training.pt', )
    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    return dataloader

def get_maneuver(traj):
