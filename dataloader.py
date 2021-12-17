import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import glob
import csv
import pandas as pd
import os
import numpy as np
import data.argoverse.argoverse.cord_conv as cord_conv
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
        os.makedirs(root+'argoverse/argoverse/preprocess', exist_ok=True)
        data_list = glob.glob(root+'argoverse/argoverse/raw/*.csv')
        data_num = len(data_list)
        veh_max = 0
        for i in range(data_num):
            data_tmp = pd.read_csv(data_list[i])
            veh_num = len(data_tmp.TRACK_ID[~data_tmp.TRACK_ID.duplicated()])

1
            if veh_num > veh_max:
                veh_max = veh_num
                index = i

        data_placeholder = np.empty(shape=(data_num, veh_max, 20, 2))
        data_placeholder[:] = float('nan')
        for i in range(data_num):
            data_instance = np.empty(shape=(veh_max, 50, 2))
            data_instance[:] = float('nan')

            data_single = pd.read_csv(data_list[i])
            time_steps = data_single.TIMESTAMP[~data_single.TIMESTAMP.duplicated()]
            id_list = data_single.TRACK_ID[~data_single.TRACK_ID.duplicated()]
            ego_id = data_single.TRACK_ID[data_single.OBJECT_TYPE == 'AV'].tolist()[0]
            ego_idx = data_single.TIMESTAMP.isin(time_steps) * data_single.TRACK_ID == ego_id
            ego_traj = np.transpose(np.stack([data_single.X[ego_idx].to_numpy(), data_single.Y[ego_idx].to_numpy()]))
            ego_time = data_single.TIMESTAMP[ego_idx].tolist()
            for j in range(len(ego_time)):
                idx = time_steps.tolist().index(ego_time[j])
                data_instance[0, idx, :] = ego_traj[j,:]
            target_id = data_single.TRACK_ID[data_single.OBJECT_TYPE == 'AGENT'].tolist()[0]
            target_idx = data_single.TIMESTAMP.isin(time_steps) * data_single.TRACK_ID == target_id
            target_traj = np.transpose(np.stack([data_single.X[target_idx].to_numpy(), data_single.Y[target_idx].to_numpy()]))
            target_time = data_single.TIMESTAMP[target_idx].tolist()
            for j in range(len(target_time)):
                idx = time_steps.tolist().index(target_time[j])
                data_instance[1, idx, :] = target_traj[j, :]

            sur_ids = id_list[~(id_list == ego_id) & ~(id_list == target_id)].tolist()
            for num in range(len(sur_ids)):
                sur_id = sur_ids[num]
                sur_idx = data_single.TIMESTAMP.isin(time_steps) * data_single.TRACK_ID == sur_id
                sur_traj = np.transpose(np.stack([data_single.X[sur_idx].to_numpy(), data_single.Y[sur_idx].to_numpy()]))
                sur_time = data_single.TIMESTAMP[sur_idx].tolist()
                for j in range(len(sur_time)):
                    idx = time_steps.tolist().index(sur_time[j])
                    data_instance[2+num, idx, :] = sur_traj[j, :]

            data_transition = cord_conv.preprocess(data_instance)
            data_rotation = cord_conv.preprocess_dir(data_transition)
            data_converted = data_rotation[data_rotation[:,19,0]==0]
            data_placeholder[i,:len(data_converted),:,:] = data_converted[:,:20,:]

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    return dataloader