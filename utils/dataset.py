import os
import numpy as np
import glob

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
from PIL import Image
from torch.utils.data import IterableDataset

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

class TransitionDataset_35(Dataset):
    def __init__(self, num_files):
        self.num_files = num_files
        
    def __getitem__(self, index):
        data = np.load('../envs/data_bisim_generalize_post/data/'+str(index)+'.npy', allow_pickle=True)[1]
        # data = np.load('../envs/data_bisim_generalize_post/data/'+str(index)+'.npy', allow_pickle=True)
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

class BisimDataset(Dataset):
    def __init__(self, file_path, num_files, offset=0, noise_scale=0., balanced=False):
        self.num_files = num_files
        self.file_path = file_path
        self.offset = offset
        self.noise_scale = noise_scale
        self.balanced = balanced
        if balanced:
            self.positive_idx = np.load('../data/positive_idx.npy')
            print('positive data: ', len(self.positive_idx))
    
    def __getitem__(self, index): 
        if self.balanced:
            if index < len(self.positive_idx):
                data = np.load(os.path.join(self.file_path, 'data/')+str(self.positive_idx[index])+'.npy', allow_pickle=True)
                label = np.load(os.path.join(self.file_path, 'label/')+str(self.positive_idx[index])+'.npy', allow_pickle=True).item() # dict
            else:
                data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
                label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict
            
        else:
            data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
            label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict


        image_state, action, image_state_next = data[0]
        image_state = torch.from_numpy(image_state).float()
        image_state_next = torch.from_numpy(image_state_next).float()
        action = torch.from_numpy(action).float()
        
        # apply noise
        noise = torch.cat([torch.zeros(4, 84, 84), self.noise_scale*(torch.rand(1, 84, 84)-0.5)], dim=0)
        image_state += noise
        image_state_next += noise
        image_state = torch.clamp(image_state, torch.zeros_like(image_state), torch.ones_like(image_state))
        image_state_next = torch.clamp(image_state_next, torch.zeros_like(image_state_next), torch.ones_like(image_state_next))


        state, _, state_next = data[1]
        state = torch.from_numpy(state).float()
        state_next = torch.from_numpy(state_next).float()
        

        # a = torch.from_numpy(a).float()
        # s_next = torch.from_numpy(s_next).float()
        reward = torch.Tensor([label['step_reward']])
        cost = torch.LongTensor([label['cost']])

        return image_state, image_state_next, state, state_next, action, reward, cost
    
    
    def __len__(self):
        return self.num_files


class BisimDataset_Spurious(Dataset):
    def __init__(self, file_path, num_files, offset=0, noise_scale=0, balanced=False):
        assert isinstance(noise_scale, int), 'noise scale should be an integer'

        self.num_files = num_files
        self.file_path = file_path
        self.offset = offset
        self.noise_scale = noise_scale
        
        self.balanced = balanced
        if balanced:
            self.positive_idx = np.load('../data/positive_idx.npy')
            print('positive data: ', len(self.positive_idx))
    
    def __getitem__(self, index): 
        if self.balanced:
            if index < len(self.positive_idx):
                data = np.load(os.path.join(self.file_path, 'data/')+str(self.positive_idx[index])+'.npy', allow_pickle=True)
                label = np.load(os.path.join(self.file_path, 'label/')+str(self.positive_idx[index])+'.npy', allow_pickle=True).item() # dict
            else:
                data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
                label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict
            
        else:
            data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
            label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict


        image_state, action, image_state_next = data[0]
        image_state = torch.from_numpy(image_state).float()
        image_state_next = torch.from_numpy(image_state_next).float()
        action = torch.from_numpy(action).float()
        
        # apply spurious noise
        noise_patch = torch.zeros_like(image_state[0])
        x_start_idx, y_start_idx = torch.randint(0, 40), torch.randint(0, 40) # torch.round((action + 1.) / 2. * image_state.shape[1]).int()
        noise_patch[x_start_idx: x_start_idx+self.noise_scale, y_start_idx: y_start_idx+self.noise_scale] = 1.
        image_state[-1, :, :] += noise_patch
        image_state_next[-1, :, :] += noise_patch

        image_state = torch.clamp(image_state, torch.zeros_like(image_state), torch.ones_like(image_state))
        image_state_next = torch.clamp(image_state_next, torch.zeros_like(image_state_next), torch.ones_like(image_state_next))



        state, _, state_next = data[1]
        state = torch.from_numpy(state).float()
        state_next = torch.from_numpy(state_next).float()
        

        # a = torch.from_numpy(a).float()
        # s_next = torch.from_numpy(s_next).float()
        reward = torch.Tensor([label['step_reward']])
        cost = torch.LongTensor([label['cost']])

        return image_state, image_state_next, state, state_next, action, reward, cost
    
    
    def __len__(self):
        return self.num_files



class Dataset_EBM(Dataset):
    def __init__(self, file_path, num_files, offset=0, noise_scale=0, balanced=False, image=True):
        assert isinstance(noise_scale, int), 'noise scale should be an integer'

        self.num_files = num_files
        self.file_path = file_path
        self.offset = offset
        self.noise_scale = noise_scale
        
        self.balanced = balanced
        self.image = image
        if image: 
            self.transform = transforms.Compose([transforms.Resize((28, 28)), transforms.PILToTensor()])


    def __getitem__(self, index): 

        data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
        label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict

        if self.image: 
            image_state, action, image_state_next = data[0]
            image_state = image_state[:3].transpose(1, 2, 0)
            im = Image.fromarray(np.array(image_state*255, dtype=np.uint8))
            im = self.transform(im) / 255.
            return im
        else: 
            state, action, state_next = data[2]
            return state
                
    def __len__(self):
        return self.num_files


class BisimDataset_Fusion_Spurious(Dataset):
    def __init__(self, file_path, num_files=None, offset=0, noise_scale=0, balanced=False, image=False):
        assert isinstance(noise_scale, int), 'noise scale should be an integer'
        
        # self.num_files = num_files
        print(file_path)
        if num_files is None: 
            self.num_files = len(glob.glob(os.path.join(file_path, "label", "*")))
        else: 
            self.num_files = num_files
        print("Num Files: ", self.num_files)
        
        self.file_path = file_path
        self.offset = offset
        self.noise_scale = noise_scale
        
        self.balanced = balanced
        # if balanced:
        #     self.positive_idx = np.load('../data/positive_idx.npy')
        #     print('positive data: ', len(self.positive_idx))
    
    def __getitem__(self, index): 
        # if self.balanced:
        #     if index < len(self.positive_idx):
        #         data = np.load(os.path.join(self.file_path, 'data/')+str(self.positive_idx[index])+'.npy', allow_pickle=True)
        #         label = np.load(os.path.join(self.file_path, 'label/')+str(self.positive_idx[index])+'.npy', allow_pickle=True).item() # dict
        #     else:
        #         data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
        #         label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict
            
        # else:
        data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
        label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict

        
        image_state, action, image_state_next = data[0]
        image_state = torch.from_numpy(image_state).float()
        image_state_next = torch.from_numpy(image_state_next).float()
        action = torch.from_numpy(action).float()
        
        # apply spurious noise
        noise_patch = torch.zeros_like(image_state[0])
        # x_start_idx, y_start_idx = torch.randint(0, 40, (1,))[0], torch.randint(0, 40, (1,))[0] # torch.round((action + 1.) / 2. * image_state.shape[1]).int()
        # x_start_idx, y_start_idx = (40*(action+1)//2).int()
        # noise_patch[x_start_idx: x_start_idx+self.noise_scale, y_start_idx: y_start_idx+self.noise_scale] = 1.
        # image_state[-1, :, :] += noise_patch
        # image_state_next[-1, :, :] += noise_patch

        image_state = torch.clamp(image_state, torch.zeros_like(image_state), torch.ones_like(image_state))
        image_state_next = torch.clamp(image_state_next, torch.zeros_like(image_state_next), torch.ones_like(image_state_next))

        state, _, state_next = data[2]
        state = torch.from_numpy(state).float()
        state_next = torch.from_numpy(state_next).float()
        

        lidar_state, _, lidar_state_next = data[1]
        lidar_state = torch.from_numpy(lidar_state).float()
        lidar_state_next = torch.from_numpy(lidar_state_next).float()
        
        # a = torch.from_numpy(a).float()
        # s_next = torch.from_numpy(s_next).float()
        reward = torch.Tensor([label['step_reward']])
        cost = torch.LongTensor([label['cost']])
        
        vc_target = torch.Tensor([label['value_cost_target']])
        vr_target = torch.Tensor([label['value_reward_target']])
        
        done = torch.Tensor([label['arrive_dest'] or label['out_of_road'] or label['crash'] or label['max_step']])

        return image_state, image_state_next, lidar_state, lidar_state_next, state, state_next, action, reward, cost, vr_target, vc_target, done
    
    
    def __len__(self):
        return self.num_files


class TransitionDataset_Baselines(IterableDataset):
    def __init__(self, file_path, num_files=None, offset=0, noise_scale=0, balanced=False, image=False):
        assert isinstance(noise_scale, int), 'noise scale should be an integer'
        if num_files is None: 
            self.num_files = len(glob.glob(os.path.join(file_path, "label", "*")))
        else: 
            self.num_files = num_files
        print("Num Files: ", self.num_files)
        self.file_path = file_path
        self.offset = offset
        self.noise_scale = noise_scale
        self.use_image = image
        self.balanced = balanced
        # if balanced:
        #     self.positive_idx = np.load('../data/positive_idx.npy')
        #     print('positive data: ', len(self.positive_idx))
    
    def _get_samples(self, index): 
        # if self.balanced:
        #     if index < len(self.positive_idx):
        #         data = np.load(os.path.join(self.file_path, 'data/')+str(self.positive_idx[index])+'.npy', allow_pickle=True)
        #         label = np.load(os.path.join(self.file_path, 'label/')+str(self.positive_idx[index])+'.npy', allow_pickle=True).item() # dict
        #     else:
        #         data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
        #         label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict
            
        # else:
        label = np.load(os.path.join(self.file_path, 'label/')+str(index+self.offset)+'.npy', allow_pickle=True).item() # dict
        data = np.load(os.path.join(self.file_path, 'data/')+str(index+self.offset)+'.npy', allow_pickle=True)
        
        if self.use_image: 

            
            image_state, action, image_state_next = data[0]
            image_state = torch.from_numpy(image_state).float()
            image_state_next = torch.from_numpy(image_state_next).float()
            action = torch.from_numpy(action).float()
            
            # apply spurious noise
            noise_patch = torch.zeros_like(image_state[0])
            # x_start_idx, y_start_idx = torch.randint(0, 40, (1,))[0], torch.randint(0, 40, (1,))[0] # torch.round((action + 1.) / 2. * image_state.shape[1]).int()
            x_start_idx, y_start_idx = (40*(action+1)//2).int()
            noise_patch[x_start_idx: x_start_idx+self.noise_scale, y_start_idx: y_start_idx+self.noise_scale] = 1.
            image_state[-1, :, :] += noise_patch
            image_state_next[-1, :, :] += noise_patch

            image_state = torch.clamp(image_state, torch.zeros_like(image_state), torch.ones_like(image_state))
            image_state_next = torch.clamp(image_state_next, torch.zeros_like(image_state_next), torch.ones_like(image_state_next))

        state, action, state_next = data[2]
        state = torch.from_numpy(state).float()
        state_next = torch.from_numpy(state_next).float()
        action = torch.from_numpy(action).float()

        lidar_state, _, lidar_state_next = data[1]
        lidar_state = torch.from_numpy(lidar_state).float()
        lidar_state_next = torch.from_numpy(lidar_state_next).float()
        
        # a = torch.from_numpy(a).float()
        # s_next = torch.from_numpy(s_next).float()
        reward = torch.Tensor([label['step_reward']])
        cost = torch.Tensor([label['cost_sparse']])
        
        vc_target = torch.Tensor([label['value_cost_target']])
        vr_target = torch.Tensor([label['value_reward_target']])
        
        done = torch.Tensor([label['arrive_dest'] or label['out_of_road'] or label['crash'] or label['max_step']])

        if self.use_image: 
            return image_state, image_state_next, lidar_state, lidar_state_next, state, state_next, action, reward, cost, vr_target, vc_target, done

        else: 
            return lidar_state, lidar_state_next, lidar_state, lidar_state_next, state, state_next, action, reward, cost, vr_target, vc_target, done
            
    def __iter__(self): 
        while True: 
            index = np.random.choice(self.num_files)
            yield self._get_samples(index)
    
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