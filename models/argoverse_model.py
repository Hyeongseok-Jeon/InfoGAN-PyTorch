import torch
import torch.nn as nn
import torch.nn.functional as F
from fractions import gcd
from torch import Tensor

"""
Architecture based on InfoGAN paper.
"""


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        '''
        norm = "GN"
        ng = 1

        n_actor = 128

        self.linear_res = LinearRes(n_actor, n_actor, norm=norm, ng=ng)
        self.linear = nn.Linear(n_actor, 2 * 20)
        '''
        #input : (batch_num, 128, 1)
        self.conv1_1 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.ConvTranspose1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.ConvTranspose1d(in_channels=128, out_channels=32, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(32)
        #input : (batch_num, 32, 4)

        self.conv3 = nn.ConvTranspose1d(in_channels=32, out_channels=8, kernel_size=4, stride=4, padding=0)
        self.bn3 = nn.BatchNorm1d(8)
        #input : (batch_num, 8, 16)

        self.conv4 = nn.ConvTranspose1d(in_channels=8, out_channels=2, kernel_size=8, stride=2, padding=2)
        self.bn4 = nn.BatchNorm1d(2)
        #input : (batch_num, 2, 34)
        self.conv5 = nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=9, stride=1, padding=1)
        #input : (batch_num, 1, 40)


    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.bn1(self.conv1_2(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.tanh(self.conv5(x))
        traj = torch.transpose(torch.cat((x[:,:,:20], x[:,:,20:]),dim=1), 1, 2)
        return traj


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        #input : (batch_num, 2, 20)
        self.Conv1d_1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=5, stride=1, padding=2, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm1d(8)

        #input : (batch_num, 8, 20)
        self.Conv1d_2 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=5, stride=2, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.bn2 = nn.BatchNorm1d(32)

        #input : (batch_num, 32, 8)
        self.Conv1d_3 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.bn3 = nn.BatchNorm1d(128)

        #input : (batch_num, 128, 2)
        self.Conv1d_4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=1, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.bn4 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = F.leaky_relu(self.bn1(self.Conv1d_1(x)))
        x = F.leaky_relu(self.bn2(self.Conv1d_2(x)))
        x = F.leaky_relu(self.bn3(self.Conv1d_3(x)))
        x = F.leaky_relu(self.bn4(self.Conv1d_4(x)))
        x = torch.flatten(torch.transpose(x, 1, 2), start_dim = 1, end_dim= -1)
        return x


class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv1d(256, 1, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, axis=-1)
        output = torch.sigmoid(self.conv(x))

        return output


class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv_disc = nn.Conv1d(128, 5, 1)
        self.conv_mu = nn.Conv1d(128, 2, 1)
        self.conv_var = nn.Conv1d(128, 2, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, axis=-1)

        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var


class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out


class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts
