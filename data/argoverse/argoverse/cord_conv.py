import numpy as np

def preprocess_dir(data):
    k = len(data)
    rotation_beta = np.empty([k])
    rot_data = np.empty([k, 20, 2])
    for i in range(k):
        dir = data[i, 4, :] - data[i, 0, :]
        rotation_beta[i] = np.arctan2(dir[1], dir[0])
        c, s = np.cos(-rotation_beta[i]), np.sin(-rotation_beta[i])
        R = np.array(((c, -s), (s, c)))
        for j in range(20):
            rot_data[i, j, :] = np.matmul(R, data[i, j, :])
    return rot_data

def preprocess(data):
    k = len(data)
    trans_data = np.empty([k, 20, 2])
    for i in range(k):
        median = data[i, 0, :].copy()
        for j in range(20):
            trans_data[i, j, :] = data[i, j, :] - median

    return trans_data

def preprocess_traj_dir(data_target, data_traj):
    k = len(data_target)
    rotation_beta = np.empty([k])
    rot_data = np.empty([k, 6, 30, 2])
    for i in range(k):
        dir = data_target[i, 20, 0:2] - data_target[i, 19, 0:2]
        rotation_beta[i] = np.arctan2(dir[1], dir[0])
        rotation_beta[i] = np.pi / 2. - rotation_beta[i]
        c, s = np.cos(rotation_beta[i]), np.sin(rotation_beta[i])
        R = np.array(((c, -s), (s, c)))
        for t in range(6):
            for j in range(30):
                rot_data[i, t, j, :] = np.matmul(R, data_traj[i, t, j, :])

    return rot_data


def preprocess_traj(data_target_org, data_traj):
    k = len(data_target_org)
    trans_data = np.empty([k, 6, 30, 2])
    for i in range(k):
        median = data_target_org[i, 19, 0:2].copy()
        for t in range(6):
            for j in range(30):
                if data_traj[i, t, j, 0] == 0 or data_traj[i, t, j, 1] == 0:
                    trans_data[i, t, j, :] = data_traj[i, t, j, :]
                else:
                    trans_data[i, t, j, :] = data_traj[i, t, j, :] - median

    return trans_data