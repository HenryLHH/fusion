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

from fusion.agent.bc.encoder import ImageStateEncoder, ImageStateEncoder_NonCausal, StateEncoder
from utils.utils import CUDA

class BNN_Agent(nn.Module):
    def __init__(self, hidden_dim=64, action_dim=2) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.image_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=self.action_dim)
        self.state_encoder = StateEncoder(state_dim=35+240, hidden_dim=hidden_dim, output_dim=self.action_dim)

        # self.lidar_encoder = StateEncoder(state_dim=240, hidden_dim=hidden_dim, output_dim=self.action_dim)
        self.criteria = nn.MSELoss()
        self.min_std, self.max_std = 1e-1, 1.

    def forward(self, state_input, action_expert, deterministic=False):
        state, lidar, img = state_input
        batch_size = state.shape[0]

        z_img = self.image_encoder(img)
        mu_img = z_img[:, :self.action_dim]
        std_img = self.min_std + (self.max_std-self.min_std)*torch.sigmoid(z_img[:, self.action_dim:])
        dist_img = D.Normal(mu_img, std_img)
        
        
        state = torch.cat([state, lidar], axis=-1)
        z_state = self.state_encoder(state)
        mu_state = z_state[:, :self.action_dim]
        std_state = self.min_std + (self.max_std-self.min_std)*torch.sigmoid(z_state[:, self.action_dim:])
        dist_state = D.Normal(mu_state, std_state)

        if not deterministic:
            act_img = dist_img.rsample()
            act_state = dist_state.rsample()
            rand_idx = torch.rand(batch_size, 1) > 0.5
            idx = CUDA(torch.cat([rand_idx, rand_idx], dim=-1))
            action = torch.where(idx > 0.5, act_img, act_state)
        else:
            std_img_ = std_img.max(-1, keepdim=True).tile(1, 2)
            std_state_ = std_state.max(-1, keepdim=True).tile(1, 2)

            action = torch.where(std_img_ > std_state_, mu_state, mu_img)
        
        policy_loss = self.criteria(action, action_expert)
        # print(action.shape)
        return action, policy_loss
    
    def act(self, state_input): 
        state, lidar, img = state_input

        z_img = self.image_encoder(img)
        mu_img = z_img[:, :self.action_dim]
        std_img = self.min_std + (self.max_std-self.min_std)*torch.sigmoid(z_img[:, self.action_dim:])
        
        
        state = torch.cat([state, lidar], axis=-1)
        z_state = self.state_encoder(state)
        mu_state = z_state[:, :self.action_dim]
        std_state = self.min_std + (self.max_std-self.min_std)*torch.sigmoid(z_state[:, self.action_dim:])

        std_img_ = std_img.max(-1, keepdim=True)[0].tile(1, 2)
        std_state_ = std_state.max(-1, keepdim=True)[0].tile(1, 2)

        action = torch.where(std_img_ > std_state_, mu_state, mu_img)
        return action