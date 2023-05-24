import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


from ssr.agent.icil.icil_state import ICIL as ICIL_state
from ssr.agent.icil.icil import ICIL as ICIL_img
from ssr.agent.icil.eval_icil import evaluate_on_env

from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.dataset import BisimDataset_Fusion_Spurious, TransitionDataset_Baselines
from utils.exp_utils import make_envs



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

def eval_loss_dataloader_state(model, dataloader, verbose=0):
    model.eval()
    loss = 0
    accuracy = 0.0
    recall_all, precision_all = 0.0, 0.0

    # pbar = tqdm(total=len(dataloader))
    for data in dataloader:
        # pbar.update(1)
        img, img_next, lidar, _, state, state_next, action, reward, cost, vr_target, vc_target, _ = data
        
        img = CUDA(img)
        img_next = CUDA(img_next)
        state = CUDA(state)
        state_next = CUDA(state_next)
        action = CUDA(action)
        # reward = CUDA(reward)
        # cost = CUDA(torch.LongTensor(cost.reshape(-1)))
        # vr_target = (CUDA(vr_target)-40) / 40.
        # vc_target = CUDA(vc_target) / 10.
        
        # loss_bisim, loss_bisim_cost = 0., 0.
        if args.image: 
            policy_loss, next_state_pred_loss, next_state_energy_loss, mi_loss, mine_loss = \
                model(img, action, img_next, deterministic=True)
        
        else:
            policy_loss, next_state_pred_loss, next_state_energy_loss, mi_loss, mine_loss = \
                model(state, action, state_next, deterministic=True)
        
        loss_act = policy_loss + next_state_energy_loss #  loss_est + #  loss_est + loss_act # + loss_cls + loss_bisim + loss_bisim_cost # loss_state_est + 0.1*kl_loss + loss_bisim_cost
        loss_rep = next_state_pred_loss + mi_loss
        # loss += 0.0001 * loss_norm
        
        loss += (loss_act+loss_rep).item()
    

        loss += (loss_act).item() # (loss_est+loss_ret+loss_bisim+loss_bisim_cost).item()
        # print('train loss: {:4f} | precision: {:.4f} | recall: {:.4f} | acc: {:.4f} '.format(loss_ret.item(), precision, recall, acc), end='\r')
        print('policy loss: {:4f} | rep. loss: {:.4f} | act loss: {:.4f} | energy loss: {:.4f} | state pred loss: {:.4f}'.format(
            loss_act.item(), loss_rep.item(), policy_loss.item(), next_state_energy_loss.item(), next_state_pred_loss.item()), end='\r')
        torch.cuda.empty_cache()
    
    loss /= len(dataloader)
    recall_all /= len(dataloader)
    precision_all /= len(dataloader)
    accuracy /= len(dataloader)

    # pbar.close()

    return loss

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
    parser.add_argument("--image", type=bool, default=False, help="use image or not")


    return parser

if __name__ == '__main__':
    NUM_EPOCHS = 200
    env = SubprocVecEnv([make_envs for _ in range(16)])
    
    args = get_train_parser().parse_args()
    
    train_set = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_mixed_dynamics_post', \
                            noise_scale=0, num_files=int(NUM_FILES*0.8), balanced=True, image=args.image) # TODO: //10
    val_set = BisimDataset_Fusion_Spurious(file_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_mixed_dynamics_post', \
                            num_files=NUM_FILES-int(NUM_FILES*0.8), offset=int(NUM_FILES*0.8), noise_scale=0, image=args.image)
    
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=True, num_workers=16)
    
    if args.image: 
        model = CUDA(ICIL_img(state_dim=5, action_dim=2, hidden_dim_input=64, hidden_dim=64))
    else:
        model = CUDA(ICIL_state(state_dim=35, action_dim=2, hidden_dim_input=64, hidden_dim=64))
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
            action = CUDA(action)
            # reward = CUDA(reward)
            # cost = CUDA(torch.LongTensor(cost.reshape(-1)))
            # vr_target = (CUDA(vr_target)-40) / 40.
            # vc_target = CUDA(vc_target) / 10.
            if args.image: 
                policy_loss, next_state_pred_loss, next_state_energy_loss, mi_loss, mine_loss = \
                    model(img, action, img_next)
            else: 
                policy_loss, next_state_pred_loss, next_state_energy_loss, mi_loss, mine_loss = \
                    model(state, action, state_next)
                
            loss_act = policy_loss + next_state_energy_loss #  loss_est + #  loss_est + loss_act # + loss_cls + loss_bisim + loss_bisim_cost # loss_state_est + 0.1*kl_loss + loss_bisim_cost
            loss_rep = next_state_pred_loss + mi_loss
            # loss += 0.0001 * loss_norm
            model.policy_opt.zero_grad()
            model.rep_opt.zero_grad()
            loss_act.backward(retain_graph=True)
            loss_rep.backward()
            model.policy_opt.step()
            model.rep_opt.step()
            
            loss_train += (loss_act+loss_rep).item()
            print('update {:04d}/{:04d} | policy loss: {:4f} | rep. loss: {:.4f} | act loss: {:.4f} | energy loss: {:.4f} | state pred loss: {:.4f}'.format(
                idx, len(train_dataloader), loss_act.item(), loss_rep.item(), policy_loss.item(), next_state_energy_loss.item(), next_state_pred_loss.item()), end='\r')
        
        loss_train /= len(train_dataloader)
        print('\n')
        model.eval()

        loss_val = eval_loss_dataloader_state(model, val_dataloader)
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
            torch.save(model.state_dict(), args.model+'.pt')
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
        
        np.save('log/icil/'+args.model+'.npy', log_dict)

    env.close()