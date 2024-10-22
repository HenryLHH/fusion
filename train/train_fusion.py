import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from fusion.agent.fusion.model import (
    Fusion,
    Fusion_Structure,
)
from fusion.agent.fusion.utils import (
    FusionDataset, 
    FusionDataset_Structure, 
    permute
)
from fusion.agent.fusion.evals import (
    evaluate_on_env, 
    evaluate_on_env_structure,
    evaluate_on_env_structure_cont
)
from fusion.envs.envs import State_TopDownMetaDriveEnv

from utils.utils import CPU, CUDA
from metadrive.manager.traffic_manager import TrafficMode


from tqdm import trange
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv

NUM_EPOCHS = 200
num_updates_per_iter = 500
lr = 1e-4
wt_decay = 1e-4
warmup_steps = 10000
decay_steps = 25000

def make_envs(): 
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=False,
        random_traffic=False,
        accident_prob=0,
        distance=20,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config="TRO"), 
        traffic_density=0.2, 
        traffic_mode=TrafficMode.Trigger,
        horizon=args.horizon-1,
    )
    return State_TopDownMetaDriveEnv(config)

block_list=["S", "T", "R", "X"]

def make_envs_single(block_id=0): 
    idx = int(block_id // 4)
    block_type=block_list[idx]
    config = dict(
        environment_num=10, 
        start_seed=0, 
        frame_stack=3, 
        safe_rl_env=False,
        random_traffic=False,
        accident_prob=0,
        distance=20,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config=block_type), 
        traffic_density=0.2, 
        traffic_mode=TrafficMode.Trigger, # Hybrid,
        horizon=args.horizon-1,
    )
    return State_TopDownMetaDriveEnv(config)


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
    parser.add_argument("--dataset", type=str, default="dataset_dafault",  help='path to the dataset')
    
    parser.add_argument("--seed", type=int, default=0, help="checkpoint to load")
    parser.add_argument("--context", type=int, default=30, help='context len of DT')
    parser.add_argument("--hidden", type=int, default=64, help='hidden dim of DT')
    parser.add_argument("--num_workers", type=int, default=16, help='number of workers for parallel dataloding')
    parser.add_argument("--num_envs", type=int, default=16, help='number of workers for parallel evaluation in environments')
    
    parser.add_argument("--horizon", type=int, default=1000, help='horizon of a task')

    parser.add_argument("--decay", type=float, default=1e-4, help='weight decay of the model')
    parser.add_argument("--rtg_scale", type=float, default=40.0, help='scale of reward to go')
    parser.add_argument("--ctg_scale", type=float, default=10.0, help='scale of cost to go')
    
    parser.add_argument("--dynamics", action="store_true", help='train world dynamics')
    parser.add_argument("--value", action="store_true", help='train value predictors')
    parser.add_argument("--bisim", action="store_true", help='use bisimulation for self supervision')

    parser.add_argument("--single_env", action="store_true", help='use single block envs')
    
    parser.add_argument("--use_pretrained", action="store_true", help='use pretrained_checkpoint')
    parser.add_argument("--checkpoint", type=str, default="context_20_single_dynamics_value",  help='path to the dataset')
    
    return parser


if __name__ == '__main__':
    args = get_train_parser().parse_args()
    os.makedirs("log/", exist_ok=True)
    os.makedirs("checkpoint/", exist_ok=True)
    os.makedirs(os.path.join("checkpoint/", args.model), exist_ok=True)

    train_set = FusionDataset_Structure(dataset_path=os.path.join('./dataset', args.dataset), 
                                context_len=args.context, rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale)
    # val_set = FusionDataset_Structure(dataset_path=os.path.join('./dataset', args.dataset), 
    #                             context_len=args.context, rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale, validation=True)
    
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=args.num_workers)
    # val_dataloader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    data_iter = iter(train_dataloader)
    
    model = CUDA(Fusion_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=args.hidden, context_len=args.context, 
                                                   n_heads=1, drop_p=0.1, max_timestep=args.horizon))
    if args.use_pretrained: 
        model.load_state_dict(torch.load(os.path.join('checkpoint', args.checkpoint, 'best.pt')))
    
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
    if args.bisim: 
        bisim_encoder_param = []
        bisim_encoder_param += [p for p in model.embed_rtg.parameters()]
        bisim_encoder_param += [p for p in model.embed_ctg.parameters()]
        bisim_encoder_param += [p for p in model.embed_state.parameters()]
        bisim_encoder_param += [p for p in model.embed_action.parameters()]
        bisim_encoder_param += [p for p in model.embed_lidar.parameters()]
        bisim_encoder_param += [p for p in model.embed_agg.parameters()]
        
        optimizer_bisim = torch.optim.AdamW(
            bisim_encoder_param,
            lr=lr, 
            weight_decay=wt_decay
        )
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lambda steps: 1.0 if steps <= decay_steps else min(0.1, np.exp(-1e-4*(steps-decay_steps))) # min((steps+1)/warmup_steps, 1)
	)
    total_updates = 0
    
    log_dict = {'reward': [], 'cost': [], 'success_rate': [], 'oor_rate': [], 'crash_rate': [], 'max_step': []}

    if args.single_env: 
        env = SubprocVecEnv([lambda: make_envs_single(i) for i in range(args.num_envs)], start_method="spawn")    
    else: 
        env = SubprocVecEnv([make_envs for _ in range(args.num_envs)], start_method="spawn")
    
    best_succ = 0.
    rtg_loss, ctg_loss = CUDA(torch.tensor(0.)), CUDA(torch.tensor(0.))
    for e in range(NUM_EPOCHS): 
        log_action_losses = []
        model.train()
        for step_idx in trange(num_updates_per_iter): 
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
            rtg_target = CUDA(torch.clone(returns_to_go).detach())
            ctg_target = CUDA(torch.clone(returns_to_go_cost).detach())
            
            state_preds, action_preds, return_preds, returns_preds_cost = model.forward(timesteps, states, actions, returns_to_go, returns_to_go_cost)

            action_preds = action_preds.view(-1, 2)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, 2)[traj_mask.view(-1,) > 0]
            
            state_preds = state_preds.view(-1, 35)[traj_mask.view(-1,) > 0]
            state_target = states[0].view(-1, 35)[traj_mask.view(-1,) > 0]
            
            return_preds = return_preds.view(-1, 1)[traj_mask.view(-1,) > 0]
            rtg_target = rtg_target.view(-1, 1)[traj_mask.view(-1,) > 0]
            
            returns_preds_cost = returns_preds_cost.view(-1, 1)[traj_mask.view(-1,) > 0]
            ctg_target = ctg_target.view(-1, 1)[traj_mask.view(-1,) > 0]
            
            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')
            if args.dynamics and args.value: 
                state_loss = F.mse_loss(state_target[:, 1:], state_preds[:, :-1], reduction='mean')
                rtg_loss = F.mse_loss(rtg_target[:, 1:], return_preds[:, :-1], reduction='mean')
                ctg_loss = F.mse_loss(ctg_target[:, 1:], returns_preds_cost[:, :-1], reduction='mean')
                
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
                optimizer_value.step()
                optimizer_value_cost.step()
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

            # update transformer's tokenizer via bisimulation loss
            if args.bisim: 
                if step_idx == num_updates_per_iter - 1:
                    for __ in range(1):
                        try:
                            timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask = next(data_iter)
                        except StopIteration:
                            data_iter = iter(train_dataloader)
                            timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask = next(data_iter)
                        
                        random_idx = np.random.permutation(len(timesteps))
        
                        timesteps = CUDA(timesteps)
                        states = [CUDA(s) for s in states]
                        actions = CUDA(actions)
                        returns_to_go = CUDA(returns_to_go)
                        returns_to_go_cost = CUDA(returns_to_go_cost)
                        traj_mask = CUDA(traj_mask)
                        
                        state_preds, action_preds, return_preds, returns_preds_cost = model.forward(timesteps, states, actions, returns_to_go, returns_to_go_cost)
                        state_preds = state_preds.view(-1, 35)[traj_mask.view(-1,) > 0]
                        return_preds = return_preds.view(-1, 1)[traj_mask.view(-1,) > 0]            
                        returns_preds_cost = returns_preds_cost.view(-1, 1)[traj_mask.view(-1,) > 0]
                        
                        timesteps_pair = permute(timesteps.detach().clone(), random_idx)
                        states_pair = ([permute(s.detach().clone(), random_idx) for s in states])
                        actions_pair = permute(actions.detach().clone(), random_idx)
                        returns_to_go_pair = permute(returns_to_go.detach().clone(), random_idx)
                        returns_to_go_cost_pair = permute(returns_to_go_cost.detach().clone(), random_idx)
                        traj_mask_pair = permute(traj_mask.detach().clone(), random_idx)              
                        
                        state_preds_pair, action_preds_pair, return_preds_pair, returns_preds_cost_pair = model.forward(timesteps_pair, states_pair, actions_pair, returns_to_go_pair, returns_to_go_cost_pair)
                        state_preds_pair = state_preds_pair.view(-1, 35)[traj_mask_pair.view(-1,) > 0].detach()
                        return_preds_pair = return_preds_pair.view(-1, 1)[traj_mask_pair.view(-1,) > 0].detach()     
                        returns_preds_cost_pair = returns_preds_cost_pair.view(-1, 1)[traj_mask_pair.view(-1,) > 0].detach()

                        dist_rtg_gt = F.huber_loss(returns_to_go.detach(), returns_to_go_pair.detach(), reduction='mean')
                        dist_ctg_gt = F.huber_loss(returns_to_go_cost.detach(), returns_to_go_cost_pair.detach(), reduction='mean')
                        dist_state_gt= F.mse_loss(states[0].detach(), states_pair[0].detach(), reduction='mean')    
                        dist_gt = dist_state_gt + dist_rtg_gt + dist_ctg_gt
                        
                        dist_latent_rtg = F.huber_loss(return_preds, return_preds_pair, reduction='mean')
                        dist_latent_ctg = F.huber_loss(returns_preds_cost, returns_preds_cost_pair, reduction='mean')
                        dist_latent_state = F.huber_loss(state_preds, state_preds_pair, reduction='mean')
                        dist_latent = dist_latent_state + dist_latent_rtg + dist_latent_ctg
                        loss_bisim = F.mse_loss(dist_gt, dist_latent, reduction='mean')
                        optimizer_bisim.zero_grad()
                        loss_bisim.backward(retain_graph=True)
                        optimizer_bisim.step()
                        print("bisim loss: ", loss_bisim.detach().cpu().item())
                
        log_action_losses = np.array(log_action_losses)
        mean_action_loss = np.mean(log_action_losses[:, 0])
        mean_rtg_loss = np.mean(log_action_losses[:, 1])
        mean_ctg_loss = np.mean(log_action_losses[:, 2])
        mean_state_loss = np.mean(log_action_losses[:, 3])
        
        
        total_updates += num_updates_per_iter    
        # input("start evaluation")
        results = evaluate_on_env_structure(model, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=300, ctg_target=2., 
                                                rtg_scale=args.rtg_scale, ctg_scale=args.ctg_scale, num_eval_ep=50, max_test_ep_len=args.horizon, use_value_pred=args.value)
        
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

        np.save(os.path.join('log/', args.model+'.npy'), log_dict)
        
        if e % 10 == 0: 
            torch.save(model.state_dict(), os.path.join('checkpoint/', args.model, '{:03d}.pt'.format(e)))
        if eval_avg_succ > best_succ: 
            print('best success rate found!')
            best_succ = eval_avg_succ
            torch.save(model.state_dict(), os.path.join('checkpoint/', args.model, 'best.pt'))
    
    env.close()
