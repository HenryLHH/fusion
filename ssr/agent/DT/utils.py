import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
# from decision_transformer.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
from tqdm import trange, tqdm
from utils.utils import CPU, CUDA


def discount_cumsum(x, gamma):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t+1]
    return disc_cumsum


# def get_d4rl_normalized_score(score, env_name):
#     env_key = env_name.split('-')[0].lower()
#     assert env_key in REF_MAX_SCORE, f'no reference score for {env_key} env to calculate d4rl score'
#     return (score - REF_MIN_SCORE[env_key]) / (REF_MAX_SCORE[env_key] - REF_MIN_SCORE[env_key])


# def get_d4rl_dataset_stats(env_d4rl_name):
#     return D4RL_DATASET_STATS[env_d4rl_name]


def evaluate_on_env(model, device, context_len, env, rtg_target, ctg_target, rtg_scale, ctg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):

    eval_batch_size = 1  # required for forward pass
    
    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_succ = []
    
    state_dim = 35 # env.observation_space.shape[0]
    act_dim = 2 # env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()

    with torch.no_grad():

        for _ in trange(num_eval_ep):

            # zeros place holders
            actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                                dtype=torch.float32, device=device)
            states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                                dtype=torch.float32, device=device)
            rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)
            costs_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                                dtype=torch.float32, device=device)
          

            # init episode
            running_state = env.reset()
            running_reward = 0
            running_cost = 0
            running_rtg = rtg_target / rtg_scale
            running_ctg = ctg_target / ctg_scale
            

            for t in range(max_test_ep_len):

                total_timesteps += 1

                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state['state']).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std
                
                # calcualate running rtg and add it in placeholder
                running_rtg = running_rtg - (running_reward / rtg_scale)
                running_ctg = running_ctg - (running_cost / ctg_scale)
                
                rewards_to_go[0, t] = running_rtg
                costs_to_go[0, t] = running_ctg

                if t < context_len:
                    _, act_preds, _, _ = model.forward(timesteps[:,:context_len],
                                                states[:,:context_len],
                                                actions[:,:context_len],
                                                rewards_to_go[:,:context_len], 
                                                costs_to_go[:, :context_len])
                    act = act_preds[0, t].detach()
                else:
                    _, act_preds, _, _ = model.forward(timesteps[:,t-context_len+1:t+1],
                                                states[:,t-context_len+1:t+1],
                                                actions[:,t-context_len+1:t+1],
                                                rewards_to_go[:,t-context_len+1:t+1], 
                                                costs_to_go[:,t-context_len+1:t+1], 
                                                )
                    act = act_preds[0, -1].detach()

                running_state, running_reward, done, info = env.step(act.cpu().numpy())
                
                # add action in placeholder
                actions[0, t] = act
                running_cost = info['cost_sparse']
                
                total_reward += running_reward
                total_cost += running_cost
                if render:
                    env.render()
                if done:
                    total_succ.append(info['arrive_dest'])
                    break

    results['eval/avg_reward'] = total_reward / num_eval_ep
    results['eval/avg_ep_len'] = total_timesteps / num_eval_ep
    results['eval/success_rate'] = np.mean(total_succ)
    results['eval/avg_cost'] = total_cost / num_eval_ep
    

    return results


def evaluate_on_env_structure(model, device, context_len, env, rtg_target, ctg_target, rtg_scale, ctg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False, use_value_pred=False):

    eval_batch_size = env.num_envs  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_succ = []
    
    state_dim = 35 # env.observation_space.shape[0]
    act_dim = 2 # env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    count_done = 0
    pbar = tqdm(total=num_eval_ep)
    timestep_last = np.zeros(eval_batch_size, dtype=np.int32)
    t = 0
    with torch.no_grad():

    # for _ in trange(num_eval_ep):
        actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                            dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                            dtype=torch.float32, device=device)
        rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=torch.float32, device=device)
        costs_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=torch.float32, device=device)
        image = torch.zeros((eval_batch_size, max_test_ep_len, 5, 84, 84),
                            dtype=torch.float32, device=device)
        lidar = torch.zeros((eval_batch_size, max_test_ep_len, 240),
                            dtype=torch.float32, device=device)
        running_state = env.reset()
        running_reward = torch.zeros((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_cost = torch.zeros((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_rtg = rtg_target / rtg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_ctg = ctg_target / ctg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        rtg_pred = rtg_target / rtg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        ctg_pred = ctg_target / ctg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        
        while count_done < num_eval_ep: 
            # print(t)
            # zeros place holders

            # init episode
            total_timesteps += eval_batch_size
            
            # add state in placeholder and normalize
            states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['state']).to(device)
            # states[:, t] = (states[:, t] - state_mean) / state_std
            image[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['img']).to(device)
            lidar[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['lidar']).to(device)
            
            # calcualate running rtg and add it in placeholder
            if use_value_pred: 
                running_rtg = torch.min(rtg_pred, running_rtg - (running_reward / rtg_scale)).clamp(torch.zeros_like(running_rtg))
                running_ctg = torch.min(ctg_pred, running_ctg - (running_cost / ctg_scale)).clamp(torch.zeros_like(running_ctg))
                
            else: 
                running_rtg = torch.clamp(running_rtg - (running_reward / rtg_scale), torch.zeros_like(running_rtg))
                running_ctg = torch.clamp(running_ctg - (running_cost / ctg_scale), torch.zeros_like(running_ctg))
                        
            rewards_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_rtg.type(torch.FloatTensor))
            costs_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_ctg.type(torch.FloatTensor))
            # print([t - timestep_last[env_id] for env_id in range(eval_batch_size)])
            
            ts_batch = torch.cat([timesteps[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else timesteps[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            state_batch = torch.cat([states[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else states[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            lidar_batch = torch.cat([lidar[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else lidar[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            img_batch = torch.cat([image[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else image[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            act_batch = torch.cat([actions[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else actions[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            rtg_batch = torch.cat([rewards_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else rewards_to_go[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            ctg_batch = torch.cat([costs_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else costs_to_go[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            
            # print(state_batch.shape)
            _, act_preds, rtg_pred, ctg_pred = model.forward(ts_batch, [state_batch, lidar_batch, img_batch], act_batch, rtg_batch, ctg_batch, deterministic=True)                        
            act = act_preds[:, -1].detach()
            rtg_pred = rtg_pred[:, -1].detach()
            ctg_pred = ctg_pred[:, -1].detach()
            # if t < context_len:
            #     _, act_preds, _, _ = model.forward(timesteps[:,:context_len],
            #                                 [states[:,:context_len], lidar[:, :context_len], image[:, :context_len]],
            #                                 actions[:,:context_len],
            #                                 rewards_to_go[:,:context_len], 
            #                                 costs_to_go[:, :context_len])
            #     act = act_preds[:, t].detach()
            # else:
            #     _, act_preds, _, _ = model.forward(timesteps[:,t-context_len+1:t+1],
            #                                 [states[:,t-context_len+1:t+1], lidar[:,t-context_len+1:t+1], image[:,t-context_len+1:t+1]],
            #                                 actions[:,t-context_len+1:t+1],
            #                                 rewards_to_go[:,t-context_len+1:t+1], 
            #                                 costs_to_go[:,t-context_len+1:t+1], 
            #                                 )
            #     act = act_preds[:, -1].detach()
            
            running_state, running_reward, done, info = env.step(act.cpu().numpy())
            # add action in placeholder
            actions[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = act
            running_cost = np.array([info[idx]['cost_sparse'] for idx in range(len(info))])
            total_reward += np.sum(running_reward)
            total_cost += running_cost.sum()
            
            running_reward = CUDA(running_reward).reshape(-1, 1)
            running_cost = CUDA(running_cost).reshape(-1, 1)
            
            if render:
                env.render()
            # print(done)
            for i in range(len(done)):
                if done[i]: 
                    total_succ.append([info[i]['arrive_dest'], info[i]['out_of_road'], info[i]['crash'], info[i]['max_step']])
                    count_done += 1
                    rewards_to_go[i] = 0
                    running_reward[i] = 0
                    running_cost[i] = 0
                    
                    actions[i] = 0 
                    states[i] = 0
                    image[i] = 0
                    lidar[i] = 0
                    running_rtg[i] = rtg_target / rtg_scale
                    running_ctg[i] = ctg_target / ctg_scale
                    timestep_last[i] = t+1
                    pbar.update(1)
                    if count_done >= num_eval_ep: 
                        break
                    # break
                # else: 
                #     if t - timestep_last[i] >= max_test_ep_len: 
                #         total_succ.append([False, False])
                #         count_done += 1
                #         rewards_to_go[i] = 0
                #         running_reward[i] = 0
                #         running_cost[i] = 0
                        
                #         actions[i] = 0 
                #         states[i] = 0
                #         image[i] = 0
                #         lidar[i] = 0
                #         running_rtg = rtg_target / rtg_scale
                #         running_ctg = ctg_target / ctg_scale
                #         timestep_last[i] = t+1
                #         pbar.update(1)
                #         if count_done >= num_eval_ep: 
                #             break
            t += 1
    
    pbar.close()
    results['eval/avg_reward'] = total_reward / count_done
    results['eval/avg_ep_len'] = total_timesteps / count_done
    results['eval/success_rate'] = np.array(total_succ)[:, 0].mean()
    results['eval/oor_rate'] = np.array(total_succ)[:, 1].mean()
    results['eval/crash_rate'] = np.array(total_succ)[:, 2].mean()
    results['eval/max_step'] = np.array(total_succ)[:, 3].mean()
    results['eval/avg_cost'] = total_cost / count_done
    

    return results


def evaluate_on_env_structure_pred(model, model_est, device, context_len, env, rtg_target, ctg_target, rtg_scale, ctg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):

    eval_batch_size = env.num_envs  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_succ = []
    
    state_dim = 35 # env.observation_space.shape[0]
    act_dim = 2 # env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    count_done = 0
    pbar = tqdm(total=num_eval_ep)
    timestep_last = np.zeros(eval_batch_size, dtype=np.int32)
    t = 0
    with torch.no_grad():

    # for _ in trange(num_eval_ep):
        actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                            dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                            dtype=torch.float32, device=device)
        rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=torch.float32, device=device)
        costs_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=torch.float32, device=device)
        image = torch.zeros((eval_batch_size, max_test_ep_len, 5, 84, 84),
                            dtype=torch.float32, device=device)
        lidar = torch.zeros((eval_batch_size, max_test_ep_len, 240),
                            dtype=torch.float32, device=device)
        running_state = env.reset()
        running_reward = torch.zeros((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_cost = torch.zeros((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_rtg = rtg_target / rtg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_ctg = ctg_target / ctg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)

        while count_done < num_eval_ep: 
            # print(t)
            # zeros place holders

            # init episode

            
            total_timesteps += eval_batch_size
            
            state_img = torch.from_numpy(running_state['img']).to(device)
            state_est = model_est.action_encoder(state_img)[1]
            states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = state_est
            # add state in placeholder and normalize
            # states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['state']).to(device)
            # states[:, t] = (states[:, t] - state_mean) / state_std
            image[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['img']).to(device)
            lidar[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['lidar']).to(device)
            
            # calcualate running rtg and add it in placeholder
            # running_rtg = torch.clamp(running_rtg - (running_reward / rtg_scale), torch.zeros_like(running_rtg))
            # running_ctg = torch.clamp(running_ctg - (running_cost / ctg_scale), torch.zeros_like(running_ctg))
            
            rewards_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_rtg.type(torch.FloatTensor))
            costs_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_ctg.type(torch.FloatTensor))
            # print([t - timestep_last[env_id] for env_id in range(eval_batch_size)])
            
            ts_batch = torch.cat([timesteps[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else timesteps[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            state_batch = torch.cat([states[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else states[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            lidar_batch = torch.cat([lidar[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else lidar[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            img_batch = torch.cat([image[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else image[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            act_batch = torch.cat([actions[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else actions[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            rtg_batch = torch.cat([rewards_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else rewards_to_go[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            ctg_batch = torch.cat([costs_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else costs_to_go[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            
            # print(state_batch.shape)
            _, act_preds, _, _ = model.forward(ts_batch, [state_batch, lidar_batch, img_batch], act_batch, rtg_batch, ctg_batch, deterministic=True)                        
            act = act_preds[:, -1].detach()
            
            # if t < context_len:
            #     _, act_preds, _, _ = model.forward(timesteps[:,:context_len],
            #                                 [states[:,:context_len], lidar[:, :context_len], image[:, :context_len]],
            #                                 actions[:,:context_len],
            #                                 rewards_to_go[:,:context_len], 
            #                                 costs_to_go[:, :context_len])
            #     act = act_preds[:, t].detach()
            # else:
            #     _, act_preds, _, _ = model.forward(timesteps[:,t-context_len+1:t+1],
            #                                 [states[:,t-context_len+1:t+1], lidar[:,t-context_len+1:t+1], image[:,t-context_len+1:t+1]],
            #                                 actions[:,t-context_len+1:t+1],
            #                                 rewards_to_go[:,t-context_len+1:t+1], 
            #                                 costs_to_go[:,t-context_len+1:t+1], 
            #                                 )
            #     act = act_preds[:, -1].detach()
            
            running_state, running_reward, done, info = env.step(act.cpu().numpy())
            # add action in placeholder
            actions[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = act
            running_cost = np.array([info[idx]['cost_sparse'] for idx in range(len(info))])
            total_reward += np.sum(running_reward)
            total_cost += running_cost.sum()
            
            running_reward = CUDA(running_reward).reshape(-1, 1)
            running_cost = CUDA(running_cost).reshape(-1, 1)
            
            if render:
                env.render()
            # print(done)
            for i in range(len(done)):
                if done[i]: 
                    total_succ.append([info[i]['arrive_dest'], info[i]['out_of_road'], info[i]['crash'], info[i]['max_step']])
                    count_done += 1
                    rewards_to_go[i] = 0
                    running_reward[i] = 0
                    running_cost[i] = 0
                    
                    actions[i] = 0 
                    states[i] = 0
                    image[i] = 0
                    lidar[i] = 0
                    running_rtg[i] = rtg_target / rtg_scale
                    running_ctg[i] = ctg_target / ctg_scale
                    timestep_last[i] = t+1
                    pbar.update(1)
                    if count_done >= num_eval_ep: 
                        break
                    # break
                # else: 
                #     if t - timestep_last[i] >= max_test_ep_len: 
                #         total_succ.append([False, False])
                #         count_done += 1
                #         rewards_to_go[i] = 0
                #         running_reward[i] = 0
                #         running_cost[i] = 0
                        
                #         actions[i] = 0 
                #         states[i] = 0
                #         image[i] = 0
                #         lidar[i] = 0
                #         running_rtg = rtg_target / rtg_scale
                #         running_ctg = ctg_target / ctg_scale
                #         timestep_last[i] = t+1
                #         pbar.update(1)
                #         if count_done >= num_eval_ep: 
                #             break
            t += 1
    
    pbar.close()
    results['eval/avg_reward'] = total_reward / count_done
    results['eval/avg_ep_len'] = total_timesteps / count_done
    results['eval/success_rate'] = np.array(total_succ)[:, 0].mean()
    results['eval/oor_rate'] = np.array(total_succ)[:, 1].mean()
    results['eval/crash_rate'] = np.array(total_succ)[:, 2].mean()
    results['eval/max_step'] = np.array(total_succ)[:, 3].mean()
    results['eval/avg_cost'] = total_cost / count_done
    

    return results



def render_env(model, device, context_len, env, rtg_target, ctg_target, rtg_scale, ctg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False):

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_succ = []
    
    state_dim = 35 # env.observation_space.shape[0]
    act_dim = 2 # env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    count_done = 0
    pbar = tqdm(total=num_eval_ep)
    timestep_last = np.zeros(eval_batch_size, dtype=np.int32)
    t = 0
    with torch.no_grad():

    # for _ in trange(num_eval_ep):
        actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                            dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                            dtype=torch.float32, device=device)
        rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=torch.float32, device=device)
        costs_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=torch.float32, device=device)
        image = torch.zeros((eval_batch_size, max_test_ep_len, 5, 84, 84),
                            dtype=torch.float32, device=device)
        lidar = torch.zeros((eval_batch_size, max_test_ep_len, 240),
                            dtype=torch.float32, device=device)
        running_state = env.reset()
        running_reward = torch.zeros((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_cost = torch.zeros((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_rtg = rtg_target / rtg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_ctg = ctg_target / ctg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)

        while count_done < num_eval_ep: 
            # print(t)
            # zeros place holders

            # init episode
            total_timesteps += eval_batch_size
            
            # add state in placeholder and normalize
            states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['state']).to(device)
            # states[:, t] = (states[:, t] - state_mean) / state_std
            image[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['img']).to(device)
            lidar[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['lidar']).to(device)
            
            # calcualate running rtg and add it in placeholder
            # running_rtg = torch.clamp(running_rtg - (running_reward / rtg_scale), torch.zeros_like(running_rtg))
            running_ctg = torch.clamp(running_ctg - (running_cost / ctg_scale), torch.zeros_like(running_ctg))
            
            rewards_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_rtg.type(torch.FloatTensor))
            costs_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_ctg.type(torch.FloatTensor))
            # print([t - timestep_last[env_id] for env_id in range(eval_batch_size)])
            
            ts_batch = torch.cat([timesteps[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else timesteps[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            state_batch = torch.cat([states[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else states[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            lidar_batch = torch.cat([lidar[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else lidar[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            img_batch = torch.cat([image[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else image[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            act_batch = torch.cat([actions[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else actions[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            rtg_batch = torch.cat([rewards_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else rewards_to_go[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            ctg_batch = torch.cat([costs_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else costs_to_go[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            
            # print(state_batch.shape)
            _, act_preds, _, _ = model.forward(ts_batch, [state_batch, lidar_batch, img_batch], act_batch, rtg_batch, ctg_batch, deterministic=True)                        
            act = act_preds[:, -1].detach()
            
            # if t < context_len:
            #     _, act_preds, _, _ = model.forward(timesteps[:,:context_len],
            #                                 [states[:,:context_len], lidar[:, :context_len], image[:, :context_len]],
            #                                 actions[:,:context_len],
            #                                 rewards_to_go[:,:context_len], 
            #                                 costs_to_go[:, :context_len])
            #     act = act_preds[:, t].detach()
            # else:
            #     _, act_preds, _, _ = model.forward(timesteps[:,t-context_len+1:t+1],
            #                                 [states[:,t-context_len+1:t+1], lidar[:,t-context_len+1:t+1], image[:,t-context_len+1:t+1]],
            #                                 actions[:,t-context_len+1:t+1],
            #                                 rewards_to_go[:,t-context_len+1:t+1], 
            #                                 costs_to_go[:,t-context_len+1:t+1], 
            #                                 )
            #     act = act_preds[:, -1].detach()
            
            running_state, running_reward, done, info = env.step(act[0].cpu().numpy())
            # add action in placeholder
            actions[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = act
            running_cost = np.array([info['cost_sparse']]) # np.array([info[idx]['cost_sparse'] for idx in range(len(info))])
            total_reward += running_reward # np.sum(running_reward)
            total_cost += running_cost.sum() # running_cost.sum()
            
            running_reward = CUDA(np.array([running_reward])).reshape(-1, 1)
            running_cost = CUDA(running_cost).reshape(-1, 1)
            
            if render:
                env.render(mode='top_down', film_size=(800, 800))
            # print(done)
            if done: 
                total_succ.append([info['arrive_dest'], info['out_of_road'], info['crash'], info['max_step']])
                count_done += 1
                rewards_to_go[0] = 0
                running_reward[0] = 0
                running_cost[0] = 0
                
                actions[0] = 0 
                states[0] = 0
                image[0] = 0
                lidar[0] = 0
                running_rtg[0] = rtg_target / rtg_scale
                running_ctg[0] = ctg_target / ctg_scale
                timestep_last[0] = t+1
                pbar.update(1)
                if count_done >= num_eval_ep: 
                    break
                running_state = env.reset()
            t += 1
    
    pbar.close()
    results['eval/avg_reward'] = total_reward / count_done
    results['eval/avg_ep_len'] = total_timesteps / count_done
    results['eval/success_rate'] = np.array(total_succ)[:, 0].mean()
    results['eval/oor_rate'] = np.array(total_succ)[:, 1].mean()
    results['eval/crash_rate'] = np.array(total_succ)[:, 2].mean()
    results['eval/max_step'] = np.array(total_succ)[:, 3].mean()
    results['eval/avg_cost'] = total_cost / count_done
    

    return results

def evaluate_on_env_structure_cont(model, device, context_len, env, rtg_target, ctg_target, rtg_scale, ctg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False, use_value_pred=False):

    eval_batch_size = env.num_envs  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_succ = []
    
    state_dim = 35 # env.observation_space.shape[0]
    act_dim = 2 # env.action_space.shape[0]

    if state_mean is None:
        state_mean = torch.zeros((state_dim,)).to(device)
    else:
        state_mean = torch.from_numpy(state_mean).to(device)

    if state_std is None:
        state_std = torch.ones((state_dim,)).to(device)
    else:
        state_std = torch.from_numpy(state_std).to(device)

    # same as timesteps used for training the transformer
    # also, crashes if device is passed to arange()
    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    count_done = 0
    pbar = tqdm(total=num_eval_ep)
    timestep_last = np.zeros(eval_batch_size, dtype=np.int32)
    t = 0
    with torch.no_grad():

    # for _ in trange(num_eval_ep):
        actions = torch.zeros((eval_batch_size, max_test_ep_len, act_dim),
                            dtype=torch.float32, device=device)
        states = torch.zeros((eval_batch_size, max_test_ep_len, state_dim),
                            dtype=torch.float32, device=device)
        rewards_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=torch.float32, device=device)
        costs_to_go = torch.zeros((eval_batch_size, max_test_ep_len, 1),
                            dtype=torch.float32, device=device)
        image = torch.zeros((eval_batch_size, max_test_ep_len, 5, 84, 84),
                            dtype=torch.float32, device=device)
        lidar = torch.zeros((eval_batch_size, max_test_ep_len, 240),
                            dtype=torch.float32, device=device)
        running_state = env.reset()
        running_reward = torch.zeros((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_cost = torch.zeros((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_rtg = rtg_target / rtg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        running_ctg = ctg_target / ctg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)

        while count_done < num_eval_ep: 
            # print(t)
            # zeros place holders

            # init episode

            
            total_timesteps += eval_batch_size
            
            # add state in placeholder and normalize
            states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['state']).to(device)
            # states[:, t] = (states[:, t] - state_mean) / state_std
            image[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['img']).to(device)
            lidar[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['lidar']).to(device)
            
            # calcualate running rtg and add it in placeholder
            running_rtg = torch.clamp(running_rtg - (running_reward / rtg_scale), torch.zeros_like(running_rtg))
            running_ctg = torch.clamp(running_ctg - (running_cost / ctg_scale), torch.zeros_like(running_ctg))
            
            rewards_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_rtg.type(torch.FloatTensor))
            costs_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_ctg.type(torch.FloatTensor))
            # print([t - timestep_last[env_id] for env_id in range(eval_batch_size)])
            
            ts_batch = torch.cat([timesteps[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else timesteps[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            state_batch = torch.cat([states[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else states[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            lidar_batch = torch.cat([lidar[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else lidar[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            img_batch = torch.cat([image[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else image[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            act_batch = torch.cat([actions[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else actions[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            rtg_batch = torch.cat([rewards_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else rewards_to_go[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            ctg_batch = torch.cat([costs_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else costs_to_go[[env_id], t-timestep_last[env_id]-context_len+1:t-timestep_last[env_id]+1] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            
            # print(state_batch.shape)
            _, act_preds, _, _ = model.forward(ts_batch, [state_batch, lidar_batch, img_batch], act_batch, rtg_batch, ctg_batch, deterministic=True)                        
            act = act_preds[:, -1].detach()
            
            # if t < context_len:
            #     _, act_preds, _, _ = model.forward(timesteps[:,:context_len],
            #                                 [states[:,:context_len], lidar[:, :context_len], image[:, :context_len]],
            #                                 actions[:,:context_len],
            #                                 rewards_to_go[:,:context_len], 
            #                                 costs_to_go[:, :context_len])
            #     act = act_preds[:, t].detach()
            # else:
            #     _, act_preds, _, _ = model.forward(timesteps[:,t-context_len+1:t+1],
            #                                 [states[:,t-context_len+1:t+1], lidar[:,t-context_len+1:t+1], image[:,t-context_len+1:t+1]],
            #                                 actions[:,t-context_len+1:t+1],
            #                                 rewards_to_go[:,t-context_len+1:t+1], 
            #                                 costs_to_go[:,t-context_len+1:t+1], 
            #                                 )
            #     act = act_preds[:, -1].detach()
            
            running_state, running_reward, done, info = env.step(act.cpu().numpy())
            # add action in placeholder
            actions[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = act
            running_cost = np.array([info[idx]['cost'] for idx in range(len(info))])
            total_reward += np.sum(running_reward)
            total_cost += running_cost.sum()
            
            running_reward = CUDA(running_reward).reshape(-1, 1)
            running_cost = CUDA(running_cost).reshape(-1, 1)
            
            if render:
                env.render()
            # print(done)
            for i in range(len(done)):
                if done[i]: 
                    total_succ.append([info[i]['arrive_dest'], info[i]['out_of_road'], info[i]['crash'], info[i]['max_step']])
                    count_done += 1
                    rewards_to_go[i] = 0
                    running_reward[i] = 0
                    running_cost[i] = 0
                    
                    actions[i] = 0 
                    states[i] = 0
                    image[i] = 0
                    lidar[i] = 0
                    running_rtg[i] = rtg_target / rtg_scale
                    running_ctg[i] = ctg_target / ctg_scale
                    timestep_last[i] = t+1
                    pbar.update(1)
                    if count_done >= num_eval_ep: 
                        break
                    # break
                # else: 
                #     if t - timestep_last[i] >= max_test_ep_len: 
                #         total_succ.append([False, False])
                #         count_done += 1
                #         rewards_to_go[i] = 0
                #         running_reward[i] = 0
                #         running_cost[i] = 0
                        
                #         actions[i] = 0 
                #         states[i] = 0
                #         image[i] = 0
                #         lidar[i] = 0
                #         running_rtg = rtg_target / rtg_scale
                #         running_ctg = ctg_target / ctg_scale
                #         timestep_last[i] = t+1
                #         pbar.update(1)
                #         if count_done >= num_eval_ep: 
                #             break
            t += 1
    
    pbar.close()
    results['eval/avg_reward'] = total_reward / count_done
    results['eval/avg_ep_len'] = total_timesteps / count_done
    results['eval/success_rate'] = np.array(total_succ)[:, 0].mean()
    results['eval/oor_rate'] = np.array(total_succ)[:, 1].mean()
    results['eval/crash_rate'] = np.array(total_succ)[:, 2].mean()
    results['eval/max_step'] = np.array(total_succ)[:, 3].mean()
    results['eval/avg_cost'] = total_cost / count_done
    

    return results


class DTTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, num_traj, context_len, rtg_scale=40.0, ctg_scale=10.0):
        self.dataset_path = dataset_path
        self.context_len = context_len
        self.num_traj = num_traj
        self.rtg_scale = rtg_scale
        self.ctg_scale = ctg_scale
        
        # # load dataset
        # with open(dataset_path, 'rb') as f:
        #     self.trajectories = pickle.load(f)

        # # calculate min len of traj, state mean and variance
        # # and returns_to_go for all traj
        # min_len = 10**6
        # states = []
        # for traj in self.trajectories:
        #     traj_len = traj['observations'].shape[0]
        #     min_len = min(min_len, traj_len)
        #     states.append(traj['observations'])
        #     # calculate returns to go and rescale them
        #     traj['returns_to_go'] = discount_cumsum(traj['rewards'], 1.0) / rtg_scale

        # # used for input normalization
        # states = np.concatenate(states, axis=0)
        # self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # # normalize states
        # for traj in self.trajectories:
        #     traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
        self.trajectories = []
        for idx in trange(num_traj):
            data = np.load(self.dataset_path + '/data/' + str(idx) + '.npy', allow_pickle=True)
            info = np.load(self.dataset_path + '/label/' + str(idx) + '.pkl', allow_pickle=True)
            traj_len = len(data)
            
            traj = {}
            traj['reward'] = np.array([[d['step_reward']] for d in info], dtype=np.float32)
            traj['cost'] = np.array([[d['cost_sparse']] for d in info],  dtype=np.float32) # cost
            traj['state'] = np.array([d['true_state'] for d in info],  dtype=np.float32)        
            traj['lidar_state'] = np.array([d['lidar_state'] for d in info], dtype=np.float32)
            traj['actions'] = np.array([d['raw_action'] for d in info], dtype=np.float32)
            traj['returns_to_go'] = discount_cumsum(traj['reward'], 1.0) / self.rtg_scale
            traj['returns_to_go_cost'] = discount_cumsum(traj['cost'], 1.0) / self.ctg_scale
            self.trajectories.append(traj)
        
        print('Trajectory loaded: ', len(self.trajectories))
    
    def __len__(self):
        return self.num_traj

    def __getitem__(self, idx):
        
        traj = self.trajectories[idx]
        traj_len = len(traj['reward'])

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['state'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            returns_to_go_cost = torch.from_numpy(traj['returns_to_go_cost'][si : si + self.context_len])
            
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)
        
        # else:
        #     padding_len = self.context_len - traj_len

        #     # padding with zeros
        #     states = torch.from_numpy(traj['observations'])
        #     states = torch.cat([states,
        #                         torch.zeros(([padding_len] + list(states.shape[1:])),
        #                         dtype=states.dtype)],
        #                        dim=0)

        #     actions = torch.from_numpy(traj['actions'])
        #     actions = torch.cat([actions,
        #                         torch.zeros(([padding_len] + list(actions.shape[1:])),
        #                         dtype=actions.dtype)],
        #                        dim=0)

        #     returns_to_go = torch.from_numpy(traj['returns_to_go'])
        #     returns_to_go = torch.cat([returns_to_go,
        #                         torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
        #                         dtype=returns_to_go.dtype)],
        #                        dim=0)

        #     timesteps = torch.arange(start=0, end=self.context_len, step=1)

        #     traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
        #                            torch.zeros(padding_len, dtype=torch.long)],
        #                           dim=0)

        return  timesteps, states, actions, returns_to_go, traj_mask



class SafeDTTrajectoryDataset(Dataset):
    def __init__(self, dataset_path, num_traj, context_len, rtg_scale=40.0, ctg_scale=10.0):
        self.dataset_path = dataset_path
        self.context_len = context_len
        self.num_traj = num_traj
        self.rtg_scale = rtg_scale
        self.ctg_scale = ctg_scale
        
        self.trajectories = []
        for idx in trange(num_traj):
            data = np.load(self.dataset_path + '/data/' + str(idx) + '.npy', allow_pickle=True)
            info = np.load(self.dataset_path + '/label/' + str(idx) + '.pkl', allow_pickle=True)
            traj_len = len(data)
            
            traj = {}
            traj['reward'] = np.array([[d['step_reward']] for d in info], dtype=np.float32)
            traj['cost'] = np.array([[d['cost_sparse']] for d in info],  dtype=np.float32) # cost
            traj['state'] = np.array([d['true_state'] for d in info],  dtype=np.float32)        
            traj['lidar_state'] = np.array([d['lidar_state'] for d in info], dtype=np.float32)
            traj['actions'] = np.array([d['raw_action'] for d in info], dtype=np.float32)
            traj['returns_to_go'] = discount_cumsum(traj['reward'], 1.0) / self.rtg_scale
            traj['returns_to_go_cost'] = discount_cumsum(traj['cost'], 1.0) / self.ctg_scale
            self.trajectories.append(traj)
        
        print('Trajectory loaded: ', len(self.trajectories))
    
    def __len__(self):
        return self.num_traj

    def __getitem__(self, idx):
        
        traj = self.trajectories[idx]
        traj_len = len(traj['reward'])

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['state'][si : si + self.context_len])
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            returns_to_go_cost = torch.from_numpy(traj['returns_to_go_cost'][si : si + self.context_len])
            
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        return  timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask



class SafeDTTrajectoryDataset_Structure(Dataset):
    def __init__(self, dataset_path, num_traj, context_len, rtg_scale=40.0, ctg_scale=10.0):
        self.dataset_path = dataset_path
        self.context_len = context_len
        self.num_traj = num_traj
        self.rtg_scale = rtg_scale
        self.ctg_scale = ctg_scale
        
        self.trajectories = []
        success_rate = []
        cost, reward = [], []
        action = []
        for idx in trange(num_traj):
            # data = np.load(self.dataset_path + '/data/' + str(idx) + '.npy', allow_pickle=True)
            info = np.load(self.dataset_path + '/label/' + str(idx) + '.pkl', allow_pickle=True)
                        
            traj = {}
            traj['reward'] = np.array([[d['step_reward']] for d in info], dtype=np.float32)
            traj['cost'] = np.array([[d['cost_sparse']] for d in info],  dtype=np.float32) # cost_sparse
            traj['state'] = np.array([d['true_state'] for d in info],  dtype=np.float32)        
            traj['lidar_state'] = np.array([d['lidar_state'] for d in info], dtype=np.float32)
            traj['img_state'] = np.zeros_like(traj['state']) # np.array(data, dtype=np.float32)
            
            traj['actions'] = np.array([d['raw_action'] for d in info], dtype=np.float32)
            traj['returns_to_go'] = discount_cumsum(traj['reward'], 1.0) / self.rtg_scale
            traj['returns_to_go_cost'] = discount_cumsum(traj['cost'], 1.0) / self.ctg_scale
            self.trajectories.append(traj)
            action.append(traj['actions'])
            success_rate.append(info[-1]['arrive_dest'])
            reward.append(traj['returns_to_go'][0])
            cost.append(traj['returns_to_go_cost'][0])
        
        print('Trajectory loaded: ', len(self.trajectories))
        print('Expert Stats: ', np.mean(success_rate), self.rtg_scale*np.mean(reward), self.ctg_scale*np.mean(cost))
        action = np.concatenate(action, axis=0).reshape(-1, 2)
        import matplotlib.pyplot as plt
        plt.scatter(action[:, 0], action[:, 1], s=1)
        plt.savefig('dataset')
        print(self.rtg_scale*np.std(reward), self.ctg_scale*np.std(cost))
        
        
    def __len__(self):
        return self.num_traj

    def __getitem__(self, idx):
        
        traj = self.trajectories[idx]
        traj_len = len(traj['reward'])

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = (torch.from_numpy(traj['state'][si : si + self.context_len]), 
                      torch.from_numpy(traj['lidar_state'][si : si + self.context_len]), 
                      torch.from_numpy(traj['img_state'][si : si + self.context_len]))
            
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            returns_to_go_cost = torch.from_numpy(traj['returns_to_go_cost'][si : si + self.context_len])
            
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        return  timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask


class SafeDTTrajectoryDataset_Structure_Cont(Dataset):
    def __init__(self, dataset_path, num_traj, context_len, rtg_scale=40.0, ctg_scale=10.0):
        self.dataset_path = dataset_path
        self.context_len = context_len
        self.num_traj = num_traj
        self.rtg_scale = rtg_scale
        self.ctg_scale = ctg_scale
        
        self.trajectories = []
        success_rate = []
        cost, reward = [], []
        action = []
        for idx in trange(num_traj):
            # data = np.load(self.dataset_path + '/data/' + str(idx) + '.npy', allow_pickle=True)
            info = np.load(self.dataset_path + '/label/' + str(idx) + '.pkl', allow_pickle=True)
                        
            traj = {}
            traj['reward'] = np.array([[d['step_reward']] for d in info], dtype=np.float32)
            traj['cost'] = np.array([[d['cost']] for d in info],  dtype=np.float32) # cost_sparse
            traj['state'] = np.array([d['true_state'] for d in info],  dtype=np.float32)        
            traj['lidar_state'] = np.array([d['lidar_state'] for d in info], dtype=np.float32)
            traj['img_state'] = np.zeros_like(traj['state']) # np.array(data, dtype=np.float32)
            
            traj['actions'] = np.array([d['raw_action'] for d in info], dtype=np.float32)
            traj['returns_to_go'] = discount_cumsum(traj['reward'], 1.0) / self.rtg_scale
            traj['returns_to_go_cost'] = discount_cumsum(traj['cost'], 1.0) / self.ctg_scale
            self.trajectories.append(traj)
            action.append(traj['actions'])
            success_rate.append(info[-1]['arrive_dest'])
            reward.append(traj['returns_to_go'][0])
            cost.append(traj['returns_to_go_cost'][0])
        
        print('Trajectory loaded: ', len(self.trajectories))
        print('Expert Stats: ', np.mean(success_rate), self.rtg_scale*np.mean(reward), self.ctg_scale*np.mean(cost))
        action = np.concatenate(action, axis=0).reshape(-1, 2)
        import matplotlib.pyplot as plt
        plt.scatter(action[:, 0], action[:, 1], s=1)
        plt.savefig('dataset')
        print(self.rtg_scale*np.std(reward), self.ctg_scale*np.std(cost))
        
        
    def __len__(self):
        return self.num_traj

    def __getitem__(self, idx):
        
        traj = self.trajectories[idx]
        traj_len = len(traj['reward'])

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = (torch.from_numpy(traj['state'][si : si + self.context_len]), 
                      torch.from_numpy(traj['lidar_state'][si : si + self.context_len]), 
                      torch.from_numpy(traj['img_state'][si : si + self.context_len]))
            
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len])
            returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            returns_to_go_cost = torch.from_numpy(traj['returns_to_go_cost'][si : si + self.context_len])
            
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        return  timesteps, states, actions, returns_to_go, returns_to_go_cost, traj_mask
