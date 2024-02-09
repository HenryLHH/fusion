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

from utils.utils import CUDA


class StateEncoder(nn.Module):
    def __init__(self, state_dim=35, hidden_dim=16, output_dim=16):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            *[nn.Linear(state_dim, hidden_dim), 
              nn.ELU(), 
              nn.Linear(hidden_dim, hidden_dim),
              nn.ELU(),
              nn.Linear(hidden_dim, output_dim)]
        )
    
    def forward(self, s):
        output = self.encoder(s)

        return output
    
class GSA_Agent(nn.Module):
    def __init__(self, hidden_dim=64, action_dim=2, K=5) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.K = K
        # self.image_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=5, output_dim=self.action_dim)
        self.state_encoder_struct = StateEncoder(state_dim=35+240, hidden_dim=hidden_dim, output_dim=K)
        self.cluster_weight = nn.Parameter(0.01*torch.randn(K, 35+240, 2))
        self.cluster_bias = nn.Parameter(self._spread_init_bias(K))
        self.cluster_std = nn.Parameter(torch.zeros(K, 2))

        # self.lidar_encoder = StateEncoder(state_dim=240, hidden_dim=hidden_dim, output_dim=self.action_dim)
        self.criteria = nn.MSELoss()
        self.min_std, self.max_std = 1e-1, 1.
        self.activate_fn = nn.Softmax()
    
    def _spread_init_bias(self, K): 
        delta = torch.tensor([2 * torch.pi / K])
        radius = 0.3
        centroid = []
        for i in range(K): 
            x = radius * torch.cos(delta*i)[0]
            y = radius * torch.sin(delta*i)[0]
            centroid.append(torch.tensor([[x, y]]))
        bias_init = CUDA(torch.cat(centroid, dim=0))
        return bias_init

    def forward(self, state_input, action_expert, deterministic=False):
        state, lidar, _ = state_input
        batch_size = state.shape[0]
        
        state = torch.cat([state, lidar], axis=-1)
        z_state = self.state_encoder_struct(state)
        if not deterministic: 
            dist_cluster = D.Categorical(self.activate_fn(z_state))
            cluster_id = dist_cluster.sample()
            # print(self.cluster_weight[cluster_id].shape, state.sha[pe, self.cluster_bias[cluster_id].shape)
            cluster_mean = torch.einsum("ijk, ij->ik", self.cluster_weight[cluster_id], state) + self.cluster_bias[cluster_id]
            cluster_std = self.min_std + (self.max_std-self.min_std) * torch.sigmoid(self.cluster_std[cluster_id])
            act_dist = D.Normal(cluster_mean, cluster_std)
            action = act_dist.rsample()

        else:
            cluster_id = z_state.argmax(dim=1)
            action = torch.einsum("ijk, ij->ik", self.cluster_weight[cluster_id], state) + self.cluster_bias[cluster_id]

        policy_loss = self.criteria(action, action_expert)
        # print(action.shape)
        return action, policy_loss
    
    def act(self, state_input): 
        state, lidar, _ = state_input
        batch_size = state.shape[0]
        
        state = torch.cat([state, lidar], axis=-1)
        z_state = self.state_encoder_struct(state)

        cluster_id = z_state.argmax(dim=1)
        action = torch.einsum("ijk, ij->ik", self.cluster_weight[cluster_id], state) + self.cluster_bias[cluster_id]
                
        return action