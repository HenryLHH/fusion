import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from fusion.agent.icil.icil_state import ICIL as ICIL_state
from fusion.agent.icil.icil import ICIL as ICIL_img
from fusion.agent.bnn.eval_utils import evaluate_on_env
from fusion.agent.bnn.bnn import BNN_Agent

from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.dataset import BisimDataset_Fusion_Spurious, TransitionDataset_Baselines
# from utils.exp_utils import make_envs
from metadrive.manager.traffic_manager import TrafficMode
from envs.envs import State_TopDownMetaDriveEnv


def make_envs(): 
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=True,
        random_traffic=False,
        accident_prob=0,
        distance=20,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config="TRO"), 
        traffic_density=0.2, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Trigger,
        horizon=args.horizon-1,
    )
    return State_TopDownMetaDriveEnv(config)

block_list=["S", "T", "R", "O"]

def make_envs_single(block_id=0): 
    idx = int(block_id // 4)
    block_type=block_list[idx]
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=True,
        random_traffic=False,
        accident_prob=0,
        distance=20,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config=block_type), 
        traffic_density=0.2, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Hybrid,
        horizon=args.horizon-1,
    )
    return State_TopDownMetaDriveEnv(config)




OBS_DIM = 35
NUM_FILES = 398000
def CPU(x):
    return x.detach().cpu().numpy()

def CUDA(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.cuda()

def reparameterize(mu, std):
    # std = torch.exp(logstd)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld



def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
    parser.add_argument("--dataset", type=str, default="dataset_mixed_post", help="dataset to load")

    parser.add_argument("--image", action="store_true", help="use image or not")
    parser.add_argument("--single_env", action="store_true", help='use single block envs')
    parser.add_argument("--horizon", type=int, default=1000, help='horizon of a task')


    return parser

if __name__ == '__main__':
    NUM_EPOCHS = 200
    args = get_train_parser().parse_args()
    
    if args.single_env: 
        env = SubprocVecEnv([lambda: make_envs_single(i) for i in range(16)], start_method="spawn")    
    else: 
        env = SubprocVecEnv([make_envs for _ in range(16)], start_method="spawn")
    

    os.makedirs("log/", exist_ok=True)
    os.makedirs("checkpoint/", exist_ok=True)
    # os.makedirs(os.path.join("log/", args.model), exist_ok=True)
    os.makedirs(os.path.join("checkpoint/", args.model), exist_ok=True)
    data_path = os.path.join("dataset", args.dataset)
    
    train_set = BisimDataset_Fusion_Spurious(file_path=data_path, \
                            noise_scale=0, balanced=True, image=args.image) # TODO: //10
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)
    
    model = CUDA(BNN_Agent(hidden_dim=64, action_dim=2))
    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    best_val_loss = 1e9
    best_succ_rate = -1
    log_dict = {'reward': [], 
                'cost': [], 
                'success_rate': [], 
                'oor_rate': [], 
                'crash_rate': [], 
                'max_step': []
                }


    for epoch in range(NUM_EPOCHS):
        loss_train, loss_val, loss_test = 0, 0, 0
        print('===========================')
        model.train()
        for idx, data in enumerate(train_dataloader):
            img, img_next, lidar, _, state, state_next, action, reward, cost, vr_target, vc_target, _ = data
            img = CUDA(img)
            img_next = CUDA(img_next)
            state = CUDA(state)
            state_next = CUDA(state_next)
            lidar = CUDA(lidar)

            action = CUDA(action)
            state_input = [state, lidar, img]

            action_pred, policy_loss = model.forward(state_input, action, deterministic=False)
            
            loss_act = policy_loss  #  loss_est + #  loss_est + loss_act # + loss_cls + loss_bisim + loss_bisim_cost # loss_state_est + 0.1*kl_loss + loss_bisim_cost
            optimizer.zero_grad()
            loss_act.backward()
            optimizer.step()

            loss_train += (loss_act).item()
            print('update {:04d}/{:04d} | policy loss: {:4f}'.format(
                idx, len(train_dataloader), loss_act.item()), end='\r')
        
        loss_train /= len(train_dataloader)
        print('\n')
        model.eval()

        # loss_val = eval_loss_dataloader_state(model, val_dataloader)
        results = evaluate_on_env(model, torch.device('cuda:0'), env, num_eval_ep=50, image=args.image)
        
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_avg_succ = results['eval/success_rate']
        eval_avg_crash = results['eval/crash_rate']
        eval_avg_oor = results['eval/oor_rate']
        eval_avg_max_step = results['eval/max_step']
        eval_avg_cost = results['eval/avg_cost']
        
        print('Epoch {:3d}, train_loss: {:4f}, val_loss:  {:4f}'.format(epoch, loss_train, loss_val))
        
        if eval_avg_succ >= best_succ_rate:
            print('best success rate find!')
            best_succ_rate = eval_avg_succ
            torch.save(model.state_dict(), os.path.join("checkpoint", args.model, 'hest.pt'))
            print('model saved!')
        
        log_str = ("=" * 60 + '\n' +
            "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' + 
            "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
            "eval avg succ: " + format(eval_avg_succ, ".5f") + '\n' +
            "eval avg crash: " + format(eval_avg_crash, ".5f") + '\n' +
            "eval avg oor: " + format(eval_avg_oor, ".5f") + '\n' +
            "eval avg overtime: " + format(eval_avg_max_step, ".5f") + '\n' +
            "eval avg cost: " + format(eval_avg_cost, ".5f") + '\n'
        )
        print(log_str)
        
        log_dict['reward'].append(eval_avg_reward)
        log_dict['cost'].append(eval_avg_cost)
        log_dict['oor_rate'].append(eval_avg_oor)
        log_dict['crash_rate'].append(eval_avg_crash)
        log_dict['max_step'].append(eval_avg_max_step)
        log_dict['success_rate'].append(eval_avg_succ)
        
        np.save(os.path.join('log', args.model+'.npy'), log_dict)
    env.close()