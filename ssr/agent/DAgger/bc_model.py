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

from encoder import ImageStateEncoder, ImageStateEncoder_NonCausal, StateEncoder
from dataset import BC_Dataset


class BC_Agent(nn.Module):
    def __init__(self, hidden_dim=64, action_dim=2) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.image_encoder = ImageStateEncoder(hidden_dim=hidden_dim, nc=3, output_dim=action_dim)
        self.state_encoder = StateEncoder(hidden_dim=hidden_dim, output_dim=action_dim)
        

    def forward(self, s_image, s_state, random=False):
        _, mu_image, std_image = self.image_encoder(s_image)
        mu_state, std_state = self.state_encoder(s_state)
        
        dist_image = D.Normal(mu_image, std_image)
        dist_state = D.Normal(mu_state, std_state)
        mu = torch.cat([mu_image.unsqueeze(1), mu_state.unsqueeze(1)], dim=1)
        # std = torch.cat([std_img.unsqueeze(1), std_state.unsqueeze(1)], dim=1)
        std = torch.cat([std_image.unsqueeze(1), std_state.unsqueeze(1)], dim=1)
        batch_size = s_state.shape[0]
        # Construct Gaussian Mixture Modle in 2D consisting of 5 equally
        # weighted bivariate normal distributions
        # mix = D.Categorical(CUDA(torch.ones(batch_size, self.action_dim)))
        # comp = D.Independent(D.Normal(
        #         mu, std), 1)
        # gmm = D.MixtureSameFamily(mix, comp)
        
        if random:
            if np.random.rand() < 0.5:
                action = dist_image.rsample()
            else:
                action = dist_state.rsample()
        else:
            idx = torch.as_tensor(std_image.max(-1)[0] > std_state.max(-1)[0], dtype=torch.long)
            idx_range = torch.arange(0, batch_size, 1)
            action = mu[idx_range, idx, :]

        # print(action.shape)

        return action, std
    
