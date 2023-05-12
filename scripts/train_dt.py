import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ssr.agent.DT.model import DecisionTransformer, SafeDecisionTransformer, SafeDecisionTransformer_Structure
from ssr.agent.DT.utils import SafeDTTrajectoryDataset, SafeDTTrajectoryDataset_Structure, evaluate_on_env, evaluate_on_env_structure
from utils.utils import CPU, CUDA

from envs.envs import State_TopDownMetaDriveEnv

from tqdm import trange
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv

NUM_EPOCHS = 200
num_updates_per_iter = 500
lr = 1e-4
wt_decay = 1e-4
warmup_steps = 10000


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")

    return parser

def make_envs(): 
    config = dict(traffic_density=0.2, vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4)))
    return State_TopDownMetaDriveEnv(config)


if __name__ == '__main__':
    args = get_train_parser().parse_args()
    train_set = SafeDTTrajectoryDataset_Structure(dataset_path='/home/haohong/0_causal_drive/baselines_clean/envs/data_bisim_cost_continuous_xsc', 
                                num_traj=1060, context_len=30)
    
    train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=16)
    data_iter = iter(train_dataloader)

    model = CUDA(SafeDecisionTransformer_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=64, context_len=30, n_heads=4, drop_p=0.1, max_timestep=1000))
    optimizer = torch.optim.AdamW(
					model.parameters(), 
					lr=lr, 
					weight_decay=wt_decay
				)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lambda steps: min((steps+1)/warmup_steps, 1)
	)
    total_updates = 0
    
    
    log_dict = {'reward': [], 'cost': [], 'success_rate': []}
    env = SubprocVecEnv([make_envs for _ in range(16)])
    
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
            state_preds, action_preds, return_preds, returns_pred_cost = model.forward(timesteps, states, actions, returns_to_go, returns_to_go_cost)
            action_preds = action_preds.view(-1, 2)[traj_mask.view(-1,) > 0]
            action_target = action_target.view(-1, 2)[traj_mask.view(-1,) > 0]

            action_loss = F.mse_loss(action_preds, action_target, reduction='mean')
            
            optimizer.zero_grad()
            action_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
            log_action_losses.append(action_loss.detach().cpu().item())

        mean_action_loss = np.mean(log_action_losses)
        total_updates += num_updates_per_iter    

    
        results = evaluate_on_env_structure(model, torch.device('cuda:0'), context_len=30, env=env, rtg_target=300, ctg_target=10, 
                                            rtg_scale=40.0, ctg_scale=10.0, num_eval_ep=16, max_test_ep_len=1000)
        
        eval_avg_reward = results['eval/avg_reward']
        eval_avg_ep_len = results['eval/avg_ep_len']
        eval_avg_succ = results['eval/success_rate']
        eval_avg_cost = results['eval/avg_cost']
        
        log_str = ("=" * 60 + '\n' +
            "num of updates: " + str(total_updates) + '\n' +
            "action loss: " +  format(mean_action_loss, ".5f") + '\n' + 
            "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' + 
			"eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
			"eval avg succ: " + format(eval_avg_succ, ".5f") + '\n' +
			"eval avg cost: " + format(eval_avg_cost, ".5f") + '\n'
            )
        
        print(log_str)
        
        log_dict['reward'].append(eval_avg_reward)
        log_dict['cost'].append(eval_avg_cost)
        log_dict['success_rate'].append(eval_avg_succ)
        if e % 1 == 0: 
            np.save('log/'+args.model+'.npy', log_dict)