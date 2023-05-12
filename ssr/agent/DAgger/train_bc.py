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

from transition_dynamics.dataset import BC_Dataset

from .bc_model import BC_Agent

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  

def CPU(x):
    return x.detach().cpu().numpy()

def CUDA(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.cuda()

def eval_loss_dataloader(model, criterion, dataloader, verbose=0):
    
    loss = 0
    pbar = tqdm(total=len(dataloader))
    
    for data in dataloader: 
        pbar.update(1)
        image, state, action = data
        image = Variable(CUDA(image))
        state = Variable(CUDA(state))
        action = CUDA(action)
        action_pred, std = agent.forward(image, state, random=False)
        loss += criterion(action_pred, action).item()
    pbar.close()
    loss /= len(dataloader)
    if verbose: 
        print('truth: ', CPU(action[0]))
        print('pred: ', CPU(action_pred[0]))
        print('var: ', CPU(std[0]))
    return loss




if __name__ == '__main__':
    print(ROOT)
    
    dataset = BC_Dataset(file_path='./envs/data_bisim_generalize_post', num_files=40_0000)
    num_training = int(len(dataset)*0.9)
    num_testing = len(dataset) - num_training
    train_set, val_set = torch.utils.data.random_split(dataset, [num_training, num_testing])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=32)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=32)

    criterion = nn.MSELoss(reduction='mean')
    agent = BC_Agent(hidden_dim=64, action_dim=2)
    agent = CUDA(agent)
    optimizer = optim.Adam(agent.parameters(), lr=1e-4, betas=(0.9, 0.999))

    best_val_loss = 1e9
    for epoch in range(200):
        loss_train, loss_val = 0, 0
        print('===========================')
        pbar = tqdm(total=len(train_loader))
        for data in train_loader:
            pbar.update(1)
            image, state, action = data
            image = Variable(CUDA(image))
            state = Variable(CUDA(state))
            action = CUDA(action)

            action_pred, _ = agent.forward(image, state, random=True)
            loss = criterion(action_pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train /= len(train_loader)

        pbar.close()
        agent.eval()
        loss_val = eval_loss_dataloader(agent, criterion, val_loader, verbose=1)
        print('Epoch {:3d}, train_loss: {:4f}, val loss:  {:4f}'.format(epoch, loss_train, loss_val))

        if loss_val < best_val_loss:
            print('best val loss find!')
            best_val_loss = loss_val
            torch.save(agent.state_dict(), os.path.join(ROOT, 'agent_bc.pt'))
            print('model saved!')