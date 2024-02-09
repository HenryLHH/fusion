import os
import numpy as np

from typing import List


import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torchvision import transforms

class TransitionDataset(Dataset):
    def __init__(self, num_files):
        self.num_files = num_files
        
    def __getitem__(self, index):
        data = np.load('../envs/data_bisim_vector/data/'+str(index)+'.npy', allow_pickle=True)
        s, a, s_next = data
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s_next = torch.from_numpy(s_next).float()
        

        return s, a, s_next
    
    def __len__(self):
        return self.num_files


class ImageTransitionDataset(Dataset):
    def __init__(self, file_path, num_files, offset=0, noise_scale=0.):
        self.num_files = num_files
        self.file_path = file_path
        self.offset = offset
        self.noise_scale = noise_scale
        
    def __getitem__(self, index):
        data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)[0]
        label = np.load(os.path.join(self.file_path, 'label/')+str(index+1+self.offset)+'.npy', allow_pickle=True)[-1] # TODO: index+1...
        noise = torch.cat([torch.zeros(2, 84, 84), self.noise_scale*(torch.rand(1, 84, 84)-0.5)], dim=0)
        # noise = self.noise_scale*(torch.rand(3, 84, 84)-0.5)
        image_state = torch.from_numpy(data).float().permute(2, 0, 1)[:3, :, :]
        image_state += noise
        # a = torch.from_numpy(a).float()

        # s_next = torch.from_numpy(s_next).float()
        true_state = torch.from_numpy(label)
        return image_state, true_state
    
    
    def __len__(self):
        return self.num_files

class ImageTransitionDataset_Gen(Dataset):
    def __init__(self, file_path, num_files, offset=0, noise_scale=0.):
        self.num_files = num_files
        self.file_path = file_path
        self.offset = offset
        self.noise_scale = noise_scale
    
    def __getitem__(self, index):
        data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)[0]
        label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True)[-1]
        # print(data)
        noise = torch.cat([torch.zeros(2, 84, 84), self.noise_scale*(torch.rand(1, 84, 84)-0.5)], dim=0)
        # noise = self.noise_scale*(torch.rand(3, 84, 84)-0.5)

        image_state = torch.from_numpy(data).float()[:3, :, :] # .permute(2, 0, 1)
        image_state += noise
        # a = torch.from_numpy(a).float()
        # s_next = torch.from_numpy(s_next).float()
        true_state = torch.from_numpy(label)[:-16]
        return image_state, true_state
    
    
    def __len__(self):
        return self.num_files



class BC_Dataset(Dataset):
    def __init__(self, file_path, num_files, offset=0, noise_scale=0.):
        self.num_files = num_files
        self.file_path = file_path
        self.offset = offset
        self.noise_scale = noise_scale
    
    def __getitem__(self, index):
        data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)

        label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True)[-1]
        # print(data)
        noise = torch.cat([torch.zeros(2, 84, 84), self.noise_scale*(torch.rand(1, 84, 84)-0.5)], dim=0)
        # noise = self.noise_scale*(torch.rand(3, 84, 84)-0.5)

        image_state = torch.from_numpy(data[0]).float()[:3, :, :] # .permute(2, 0, 1)
        image_state += noise
        action = torch.from_numpy(data[1]).float()
        action = torch.clip(action, -torch.ones(2,), torch.ones(2, ))
        true_state = torch.from_numpy(label)

        return image_state, true_state, action
    
    
    def __len__(self):
        return self.num_files