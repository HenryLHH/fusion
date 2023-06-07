import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ssr.agent.DT.model import (
    DecisionTransformer, 
    SafeDecisionTransformer, 
    SafeDecisionTransformer_Structure_Dynamics
)
from ssr.agent.DT.utils import (
    SafeDTTrajectoryDataset, 
    SafeDTTrajectoryDataset_Structure, 
    SafeDTTrajectoryDataset_Structure_Cont,
    evaluate_on_env, 
    evaluate_on_env_structure,
    evaluate_on_env_structure_cont
)
from utils.utils import CUDA, CUDA
from metadrive.manager.traffic_manager import TrafficMode

from envs.envs import State_TopDownMetaDriveEnv

from tqdm import trange
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.exp_utils import make_envs

NUM_EPOCHS = 200
num_updates_per_iter = 500
lr = 1e-4
wt_decay = 1e-4
warmup_steps = 10000
decay_steps = 25000

def evaluate_dynamics(model, dataloader): 
    val_loss, val_state_loss, val_lidar_loss = 0., 0., 0.
    for data in dataloader: 
        timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask = data
        timesteps = CUDA(timesteps)
        states = [CUDA(s) for s in states]
        actions = CUDA(actions)
        returns_to_go = CUDA(returns_to_go)
        returns_to_go_cost = CUDA(returns_to_go_cost)
        traj_mask = CUDA(traj_mask)
        
        action_target = CUDA(torch.clone(actions).detach())
        state_preds, state_preds_lidar = model.forward(timesteps, states, actions, returns_to_go, returns_to_go_cost)

        state_loss = F.mse_loss(states[0][:, 1:], state_preds[:, :-1], reduction='mean')
        state_lidar_loss = F.mse_loss(states[1][:, 1:], state_preds_lidar[:, :-1], reduction='mean')
        
        loss = state_loss + 0.1 * state_lidar_loss
        val_loss += loss
        val_state_loss += state_loss
        val_lidar_loss += state_lidar_loss

    return val_loss, val_state_loss, val_lidar_loss

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
    parser.add_argument("--dataset", type=str, default="dataset_dafault",  help='path to the dataset')

    parser.add_argument("--seed", type=int, default=0, help="checkpoint to load")
    parser.add_argument("--context", type=int, default=30, help='context len of DT')
    parser.add_argument("--hidden", type=int, default=64, help='hidden dim of DT')
    parser.add_argument("--decay", type=float, default=1e-4, help='weight decay of the model')
    parser.add_argument("--rtg_scale", type=float, default=40.0, help='scale of reward to go')
    parser.add_argument("--ctg_scale", type=float, default=10.0, help='scale of cost to go')
    
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--dynamics", action="store_true", help='train world dynamics')
    parser.add_argument("--value", action="store_true", help='train value predictors')
    
    return parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    os.makedirs("log/", exist_ok=True)
    os.makedirs("checkpoint/", exist_ok=True)

    # os.makedirs(os.path.join("log/", args.model), exist_ok=True)
    os.makedirs(os.path.join("checkpoint/", args.model), exist_ok=True)

    if args.continuous: 
        train_set = SafeDTTrajectoryDataset_Structure_Cont(dataset_path=os.path.join('./dataset', args.dataset), 
                                    context_len=args.context, rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale)
        val_set = SafeDTTrajectoryDataset_Structure_Cont(dataset_path=os.path.join('./dataset', args.dataset+'_val'), 
                            context_len=args.context, rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale)
    else:
        train_set = SafeDTTrajectoryDataset_Structure(dataset_path=os.path.join('./dataset', args.dataset), 
                                    context_len=args.context, rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale)
        val_set = SafeDTTrajectoryDataset_Structure(dataset_path=os.path.join('./dataset', args.dataset+'_val'), 
                                    context_len=args.context, rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale)
        
    # train_set = SafeDTTrajectoryDataset_Structure_Cont(dataset_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_cost_continuous', 
    #                             num_traj=1056, context_len=30)
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=4)

    data_iter = iter(train_dataloader)
    
    model = CUDA(SafeDecisionTransformer_Structure_Dynamics(state_dim=35, act_dim=2, n_blocks=3, h_dim=args.hidden, context_len=args.context, 
                                                   n_heads=1, drop_p=0.1, max_timestep=1001))

    optimizer_dynamics = torch.optim.AdamW(
                    model.parameters(), 
                    lr=lr, 
                    weight_decay=wt_decay
    )

    loss_criteria = nn.GaussianNLLLoss()

    total_updates = 0
    
    log_dict = {'train_state_loss': [], 'val_state_loss': [], 
                'train_lidar_loss': [], 'val_lidar_loss': []}
    # env = SubprocVecEnv([make_envs for _ in range(16)])

    best_succ, best_loss = 0., 1e9
    rtg_loss, ctg_loss = CUDA(torch.tensor(0.)), CUDA(torch.tensor(0.))
    for e in range(NUM_EPOCHS): 
        log_action_losses = []
        model.train()
        for _ in trange(num_updates_per_iter): 
            try:
                timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask = next(data_iter)
            
            timesteps = CUDA(timesteps)
            states = [CUDA(s) for s in states]
            actions = CUDA(actions)
            returns_to_go = CUDA(returns_to_go)
            returns_to_go_cost = CUDA(returns_to_go_cost)
            traj_mask = CUDA(traj_mask)
            
            action_target = CUDA(torch.clone(actions).detach())
            state_preds, state_preds_lidar = model.forward(timesteps, states, actions, returns_to_go, returns_to_go_cost)

            state_loss = F.mse_loss(states[0][:, 1:], state_preds[:, :-1], reduction='mean')
            state_lidar_loss = F.mse_loss(states[1][:, 1:], state_preds_lidar[:, :-1], reduction='mean')
            
            loss = state_loss + 0.1 * state_lidar_loss
            optimizer_dynamics.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer_dynamics.step()

            log_action_losses.append([state_loss.detach().cpu().item(), 
                        state_lidar_loss.detach().cpu().item()])     
            
        log_action_losses = np.array(log_action_losses)
        train_state_loss = np.mean(log_action_losses[:, 0])
        train_lidar_loss = np.mean(log_action_losses[:, 1])
        
        
        total_updates += num_updates_per_iter    
        
        val_loss, val_state_loss, val_lidar_loss = evaluate_dynamics(model, val_dataloader)

        log_str = ("=" * 60 + '\n' +
            "num of updates: " + str(total_updates) + '\n' +
            "dynamics loss: " +  format(train_state_loss, ".5f") + '\n' + 
            "lidar loss: " +  format(train_lidar_loss, ".5f") + '\n' + 
            "dynamics loss val: " +  format(val_state_loss, ".5f") + '\n' + 
            "lidar loss val: " +  format(val_lidar_loss, ".5f") + '\n'
        )
        
        print(log_str)
        
        log_dict['train_state_loss'].append(train_state_loss)
        log_dict['train_lidar_loss'].append(train_lidar_loss)
        log_dict['val_state_loss'].append(val_state_loss)
        log_dict['val_lidar_loss'].append(val_lidar_loss)
        
        np.save(os.path.join('log/', args.model+'.npy'), log_dict)
         
        if e % 10 == 0: 
            torch.save(model.state_dict(), os.path.join('checkpoint/', args.model, '{:03d}.pt'.format(e)))
        if val_loss < best_loss: 
            print('best val loss found!')
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join('checkpoint/', args.model, 'best.pt'))
    
    # env.close()