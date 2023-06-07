import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch 
import torch.nn as nn
import torch.distributions as D
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim

from ssr.agent.DAgger.encoder import ImageStateEncoder, ImageStateEncoder_NonCausal, StateEncoder


class BC_Agent(nn.Module):
    def __init__(self, hidden_dim=64, action_dim=2, use_img=False) -> None:
        super().__init__()
        self.action_dim = action_dim
        if use_img: 
            self.image_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=hidden_dim//2)
        else: 
            self.state_encoder = StateEncoder(state_dim=35, hidden_dim=hidden_dim, output_dim=hidden_dim//2)
        
        self.lidar_encoder = StateEncoder(state_dim=240, hidden_dim=hidden_dim, output_dim=hidden_dim//2)
        self.use_img = use_img
        self.aggregate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim*2))
        self.criteria = nn.MSELoss()
        self.min_std, self.max_std = 1e-1, 1.

    def forward(self, state, lidar, action_expert, deterministic=False):
        batch_size = state.shape[0]

        if self.use_img: 
            z_state = self.image_encoder(state)
        else: 
            z_state = self.state_encoder(state)

        z_lidar = self.lidar_encoder(lidar)
        z_agg = torch.cat([z_state, z_lidar], axis=1)
        z_agg = self.aggregate(z_agg)

        mu_act, std_act = z_agg[:, :self.action_dim], self.min_std + (self.max_std-self.min_std)*torch.sigmoid(z_agg[:, self.action_dim:])
        dist = D.Normal(mu_act, std_act)

        if not deterministic:
            action = dist.rsample()
        else:
            # idx = torch.as_tensor(std_image.max(-1)[0] > std_state.max(-1)[0], dtype=torch.long)
            # idx_range = torch.arange(0, batch_size, 1)
            # action = mu[idx_range, idx, :]
            action = mu_act

        policy_loss = self.criteria(action, action_expert)
        # print(action.shape)
        return action, policy_loss
    
    def act(self, state, lidar): 
        if self.use_img: 
            z_state = self.image_encoder(state)
        else: 
            z_state = self.state_encoder(state)
        
        z_lidar = self.lidar_encoder(lidar)
        z_agg = torch.cat([z_state, z_lidar], axis=1)
        action = z_agg[:, :self.action_dim]

        return action