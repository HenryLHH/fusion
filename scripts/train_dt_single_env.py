import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ssr.agent.DT.model import (
    DecisionTransformer, 
    SafeDecisionTransformer, 
    SafeDecisionTransformer_Structure
)
from ssr.agent.DT.utils import (
    SafeDTTrajectoryDataset, 
    SafeDTTrajectoryDataset_Structure, 
    SafeDTTrajectoryDataset_Structure_Cont,
    evaluate_on_env, 
    evaluate_on_env_structure,
    evaluate_on_env_structure_cont
)
from utils.utils import CPU, CUDA
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


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
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
    if args.continuous: 
        train_set = SafeDTTrajectoryDataset_Structure_Cont(dataset_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_mixed_dynamics', 
                                    num_traj=959, context_len=args.context, rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale)
    else:
        train_set = SafeDTTrajectoryDataset_Structure(dataset_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_mixed_dynamics', 
                                    num_traj=959, context_len=args.context, rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale)
    # train_set = SafeDTTrajectoryDataset_Structure_Cont(dataset_path='/home/haohong/0_causal_drive/baselines_clean/data/data_bisim_cost_continuous', 
    #                             num_traj=1056, context_len=30)
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    data_iter = iter(train_dataloader)
    
    model = CUDA(SafeDecisionTransformer_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=args.hidden, context_len=args.context, 
                                                   n_heads=1, drop_p=0.1, max_timestep=1000))
    optimizer = torch.optim.AdamW(
					model.parameters(), 
					lr=lr, 
					weight_decay=wt_decay
				)
    if args.dynamics: 
        optimizer_dynamics = torch.optim.AdamW(
                        model.parameters(), 
                        lr=lr, 
                        weight_decay=wt_decay
                    )
    if args.value: 
        optimizer_value = torch.optim.AdamW(
                    model.parameters(), 
                    lr=lr, 
                    weight_decay=wt_decay
                )
        optimizer_value_cost = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=wt_decay
        )
    loss_criteria = nn.GaussianNLLLoss()
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lambda steps: 1.0 if steps <= decay_steps else min(0.1, np.exp(-1e-4*(steps-decay_steps))) # min((steps+1)/warmup_steps, 1)
	)
    total_updates = 0
    
    log_dict = {'reward': [], 'cost': [], 'success_rate': [], 'oor_rate': [], 'crash_rate': [], 'max_step': []}
    env = SubprocVecEnv([make_envs for _ in range(16)])
    best_succ = 0.
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
            state_preds, action_preds, return_preds, returns_preds_cost = model.forward(timesteps, states, actions, returns_to_go, returns_to_go_cost)
            action_preds = action_preds.view(-1, 2)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, 2)[traj_mask.view(-1,) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')
            if args.dynamics and args.value: 
                state_loss = F.mse_loss(states[0][:, 1:], state_preds[:, :-1], reduction='mean')
                rtg_loss = F.mse_loss(returns_to_go[:, 1:], return_preds[:, :-1], reduction='mean')
                ctg_loss = F.mse_loss(returns_to_go_cost[:, 1:], returns_preds_cost[:, :-1], reduction='mean')
                
                optimizer.zero_grad()
                optimizer_dynamics.zero_grad()
                optimizer_value.zero_grad()
                optimizer_value_cost.zero_grad()
                action_loss.backward(retain_graph=True)
                rtg_loss.backward(retain_graph=True)
                ctg_loss.backward(retain_graph=True)
                state_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                optimizer_dynamics.step()
                scheduler.step()
                log_action_losses.append([action_loss.detach().cpu().item(), 
                            rtg_loss.detach().cpu().item(), 
                            ctg_loss.detach().cpu().item(), 
                            state_loss.detach().cpu().item()])     
                
            elif args.dynamics:
                state_loss = F.mse_loss(states[0][:, 1:], state_preds[:, :-1], reduction='mean')
                optimizer.zero_grad()
                optimizer_dynamics.zero_grad()
                action_loss.backward(retain_graph=True)
                state_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                optimizer_dynamics.step()
                scheduler.step()
                log_action_losses.append([action_loss.detach().cpu().item(), 
                            rtg_loss.detach().cpu().item(), 
                            ctg_loss.detach().cpu().item(), 
                            state_loss.detach().cpu().item()])
            else: 
                optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()
                log_action_losses.append([action_loss.detach().cpu().item(), rtg_loss.detach().cpu().item(), ctg_loss.detach().cpu().item(), 0.])

        log_action_losses = np.array(log_action_losses)
        mean_action_loss = np.mean(log_action_losses[:, 0])
        mean_rtg_loss = np.mean(log_action_losses[:, 1])
        mean_ctg_loss = np.mean(log_action_losses[:, 2])
        mean_state_loss = np.mean(log_action_losses[:, 3])
        
        
        total_updates += num_updates_per_iter    

        if args.continuous: 
            results = evaluate_on_env_structure_cont(model, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=300, ctg_target=2., 
                                                rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale, num_eval_ep=50, max_test_ep_len=1000, use_value_pred=args.value)
        else:
            results = evaluate_on_env_structure(model, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=300, ctg_target=2., 
                                                 rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale, num_eval_ep=50, max_test_ep_len=1000, use_value_pred=args.value)
                    
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_avg_succ = results['eval/success_rate']
        eval_avg_crash = results['eval/crash_rate']
        eval_avg_oor = results['eval/oor_rate']
        eval_avg_max_step = results['eval/max_step']
        eval_avg_cost = results['eval/avg_cost']
        
        log_str = ("=" * 60 + '\n' +
            "num of updates: " + str(total_updates) + '\n' +
            "action loss: " +  format(mean_action_loss, ".5f") + '\n' + 
            "rtg loss: " +  format(mean_rtg_loss, ".5f") + '\n' + 
            "ctg loss: " +  format(mean_ctg_loss, ".5f") + '\n' + 
            "dynamics loss: " +  format(mean_state_loss, ".5f") + '\n' + 
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
        # if e % 10 == 0: 
        np.save('log/'+args.model+'.npy', log_dict)
        if e % 10 == 0: 
            torch.save(model.state_dict(), 'checkpoint/'+args.model+'_{:3d}.pt'.format(e))
        if eval_avg_succ > best_succ: 
            print('best success rate found!')
            best_succ = eval_avg_succ
            torch.save(model.state_dict(), 'checkpoint/'+args.model+'_best.pt')
    
    env.close()