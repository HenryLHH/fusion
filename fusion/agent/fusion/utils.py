import os
import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
# from decision_transformer.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
from tqdm import trange, tqdm
from utils.utils import CPU, CUDA
import glob

def permute(x, random_idx=None): 
    if random_idx is None:
        n_batch = len(x)
        random_idx = np.random.permutation(n_batch)
    return x[random_idx]

def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum

class FusionDataset(Dataset):
    def __init__(self, dataset_path, num_traj, context_len, rtg_scale=40.0, ctg_scale=10.0):
        self.dataset_path = dataset_path
        self.context_len = context_len
        self.num_traj = num_traj
        self.rtg_scale = rtg_scale
        self.ctg_scale = ctg_scale
        
        self.trajectories = []
        for idx in trange(num_traj):
            data = np.load(self.dataset_path + '/data/' + str(idx) + '.npy', allow_pickle=True)
            info = np.load(self.dataset_path + '/label/' + str(idx) + '.pkl', allow_pickle=True)
            traj_len = len(data)
            
            traj = {}
            traj['reward'] = np.array([[d['step_reward']] for d in info], dtype=np.float32)
            traj['cost'] = np.array([[d['cost']] for d in info],  dtype=np.float32) # cost
            traj['state'] = np.array([d['true_state'] for d in info],  dtype=np.float32)        
            traj['lidar_state'] = np.array([d['lidar_state'] for d in info], dtype=np.float32)
            traj['actions'] = np.array([d['raw_action'] for d in info], dtype=np.float32)
            traj['returns_to_go'] = discount_cumsum(traj['reward'], 1.0) / self.rtg_scale
            traj['returns_to_go_cost'] = discount_cumsum(traj['cost'], 1.0) / self.ctg_scale
            self.trajectories.append(traj)
        
        print('Trajectory loaded: ', len(self.trajectories))
    
    def __len__(self):
        return self.num_traj

    def __getitem__(self, idx):
        
        traj = self.trajectories[idx]
        traj_len = len(traj['reward'])

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['state'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            returns_to_go_cost = torch.from_numpy(traj['returns_to_go_cost'][si : si + self.context_len])
            
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)
        
        return  timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask



class FusionDataset_Structure(Dataset):
    def __init__(self, dataset_path, context_len, rtg_scale=40.0, ctg_scale=10.0, num_traj=None, validation=False):
        self.dataset_path = dataset_path
        if num_traj is None:
            num_traj = len(glob.glob(os.path.join(dataset_path, "label/*.pkl")))
        self.context_len = context_len
        self.num_traj = num_traj
        self.rtg_scale = rtg_scale
        self.ctg_scale = ctg_scale
        
        self.trajectories = []
        success_rate = []
        cost, reward, vel_cost = [], [], []
        action = []
        collision, oor, max_time, success = 0, 0, 0, 0

        for idx in trange(num_traj):
            info = np.load(self.dataset_path + '/label/' + str(idx) + '.pkl', allow_pickle=True)
            
            traj = {}
            traj['reward'] = np.array([[d['step_reward']] for d in info], dtype=np.float32)
            traj['cost'] = np.array([[d['cost']] for d in info],  dtype=np.float32)
            traj['state'] = np.array([d['last_state'] for d in info],  dtype=np.float32)      
            traj['lidar_state'] = np.array([d['last_lidar'] for d in info], dtype=np.float32)

            if validation:
                data = np.load(self.dataset_path + '/data/' + str(idx) + '.npy', allow_pickle=True)            
                traj['img_state'] = np.array(data, dtype=np.float32)
            else:
                traj['img_state'] = np.zeros((5, 84, 84), dtype=np.float32) # np.zeros_like(traj['state'])
                        
            traj['velocity_cost'] = np.array([[d['velocity_cost'] > 0.] for d in info],  dtype=np.float32)
            
            traj['actions'] = np.array([d['raw_action'] for d in info], dtype=np.float32)
            traj['returns_to_go'] = discount_cumsum(traj['reward'], 1.0) / self.rtg_scale
            traj['returns_to_go_cost'] = discount_cumsum(traj['cost'], 1.0) / self.ctg_scale
            # filter out traj longer than 1000
            if len(traj['cost']) > 1000:
                self.num_traj -= 1
            else: 
                self.trajectories.append(traj)
                action.append(traj['actions'])
                success_rate.append(info[-1]['arrive_dest'])
                reward.append(traj['returns_to_go'][0])
                cost.append(traj['returns_to_go_cost'][0])
                vel_cost.append(traj['velocity_cost'])

            if np.array([info[i]['crash'] for i in range(len(traj['cost']))]).any(): 
                collision += 1
            elif info[-1]['out_of_road']: 
                oor += 1
            elif info[-1]['max_step']: 
                max_time += 1
            elif info[-1]['arrive_dest']: 
                success += 1
        
        print("succ: {:.3f}, crash: {:.3f}, oor: {:.3f}, max_time: {:.3f}".format(
            success/num_traj, collision/num_traj, oor/num_traj, max_time/num_traj))
        vel_cost = np.concatenate(vel_cost, axis=0)
        print('Trajectory loaded: ', len(self.trajectories))
        print('Expert Stats: success rate: {:.2f} | reward: {:.2f} | cost: {:.2f} | overspeed: {:.2f}'.\
              format(np.mean(success_rate), self.rtg_scale*np.mean(reward), self.ctg_scale*np.mean(cost), np.mean(vel_cost)))
        # action = np.concatenate(action, axis=0).reshape(-1, 2)
        # import matplotlib.pyplot as plt
        # plt.scatter(action[:, 0], action[:, 1], s=1)
        # plt.savefig('dataset')
        # print(self.rtg_scale*np.std(reward), self.ctg_scale*np.std(cost))
        
        
    def __len__(self):
        return self.num_traj

    def __getitem__(self, idx):
        
        traj = self.trajectories[idx]
        traj_len = len(traj['reward'])
        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)
            
            states = (torch.from_numpy(traj['state'][si : si + self.context_len]), 
                      torch.from_numpy(traj['lidar_state'][si : si + self.context_len]), 
                      torch.from_numpy(traj['img_state'])[None, ...].repeat(self.context_len, 1, 1, 1))
                    #   torch.from_numpy(traj['img_state'][si : si + self.context_len]))
            
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            returns_to_go_cost = torch.from_numpy(traj['returns_to_go_cost'][si : si + self.context_len])
            
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        return  timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask
