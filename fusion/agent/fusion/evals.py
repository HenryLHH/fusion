import os
import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
# from decision_transformer.d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
from tqdm import trange, tqdm
from utils.utils import CPU, CUDA
import glob

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


def render_env(model, device, context_len, env, rtg_target, ctg_target, rtg_scale, ctg_scale,
                    num_eval_ep=10, max_test_ep_len=1000, use_value_pred=False, 
                    state_mean=None, state_std=None, render=False):
    def _get_tensor_batch(array_in, verbose=0):
        array = array_in.clone().detach()
        tensor = torch.cat([array[[env_id], :context_len] if t-timestep_last[env_id] < context_len \
                else array[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)

        if verbose:
            print([t-timestep_last[env_id] for env_id in range(eval_batch_size)])
            tensor_debug = array[-1, -context_len:].to(device)
            print('tensor ifelse: ', tensor[-1])
            print('tensor debug: ', tensor_debug)

            print(tensor.shape)        
        return tensor

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
                running_rtg = torch.max(rtg_pred, running_rtg - (running_reward / rtg_scale)).clamp(torch.zeros_like(running_rtg))
                running_ctg = torch.min(ctg_pred, running_ctg - (running_cost / ctg_scale)).clamp(torch.zeros_like(running_ctg))
                
            else: 
                running_rtg = torch.clamp(running_rtg - (running_reward / rtg_scale), torch.zeros_like(running_rtg))
                running_ctg = torch.clamp(running_ctg - (running_cost / ctg_scale), torch.zeros_like(running_ctg))
            
            rewards_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_rtg.type(torch.FloatTensor))
            costs_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_ctg.type(torch.FloatTensor))
            # print([t - timestep_last[env_id] for env_id in range(eval_batch_size)])
            
            ts_batch = _get_tensor_batch(timesteps)
            state_batch = _get_tensor_batch(states)
            lidar_batch = _get_tensor_batch(lidar)
            img_batch = _get_tensor_batch(image)
            act_batch = _get_tensor_batch(actions)
            rtg_batch = _get_tensor_batch(rewards_to_go)
            ctg_batch = _get_tensor_batch(costs_to_go)
            
            # print(state_batch.shape)
            _, act_preds, rtg_pred, ctg_pred = model.forward(ts_batch, [state_batch, lidar_batch, img_batch], act_batch, rtg_batch, ctg_batch, deterministic=True)                        
            act = act_preds[:, -1].detach()
            rtg_pred = rtg_pred[:, -1].detach()
            ctg_pred = ctg_pred[:, -1].detach()

            
            running_state, running_reward, done, info = env.step(act[0].cpu().numpy())
            # add action in placeholder
            actions[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = act
            running_cost = np.array([info['cost']]) # np.array([info[idx]['cost'] for idx in range(len(info))])
            total_reward += running_reward # np.sum(running_reward)
            total_cost += running_cost.sum() # running_cost.sum()
            
            running_reward = CUDA(np.array([running_reward])).reshape(-1, 1)
            running_cost = CUDA(running_cost).reshape(-1, 1)
            
            if render:
                env.render() # mode='top_down', film_size=(800, 800))
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
                else timesteps[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            state_batch = torch.cat([states[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else states[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            lidar_batch = torch.cat([lidar[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else lidar[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            img_batch = torch.cat([image[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else image[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            act_batch = torch.cat([actions[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else actions[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            rtg_batch = torch.cat([rewards_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else rewards_to_go[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            ctg_batch = torch.cat([costs_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else costs_to_go[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
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


def evaluate_expert(env, num_ep=50): 
    env.reset()
    count_done = 0
    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_overspeed = 0
    total_succ = []
    pbar = tqdm(total=num_ep)
    while count_done < num_ep: 
        action = np.zeros((16, 2))
        total_timesteps += 16
        # print('action: ', action)
        obs, rew, done, info = env.step(action)
        total_cost += np.sum([info[i]["cost"] for i in range(len(done))])
        running_overspeed = np.array([info[idx]['velocity_cost']>0. for idx in range(len(info))])
        total_overspeed += np.sum(running_overspeed)
        for i in range(len(done)):
            if done[i]: 
                # print(info[i].keys())
                total_reward += info[i]["episode_reward"]
                total_succ.append([info[i]['arrive_dest'], info[i]['out_of_road'], info[i]['crash'], info[i]['max_step']])
                count_done += 1
                pbar.update(1)
    
    results['eval/avg_reward'] = total_reward / count_done
    results['eval/avg_ep_len'] = total_timesteps / count_done
    results['eval/success_rate'] = np.array(total_succ)[:, 0].mean()
    results['eval/oor_rate'] = np.array(total_succ)[:, 1].mean()
    results['eval/crash_rate'] = np.array(total_succ)[:, 2].mean()
    results['eval/max_step'] = np.array(total_succ)[:, 3].mean()
    results['eval/avg_cost'] = total_cost / count_done
    results['eval/over_speed'] = total_overspeed / total_timesteps
    pbar.close()

    return results

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
                running_cost = info['cost']
                
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


def evaluate_on_env_nocost(model, device, context_len, env, rtg_target, ctg_target, rtg_scale, ctg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False, use_value_pred=False, 
                    use_state_pred=False, model_est=None):
    
    def _get_tensor_batch(array_in, verbose=0):
        array = array_in.clone().detach()
        tensor = torch.cat([array[[env_id], :context_len] if t-timestep_last[env_id] < context_len \
                else array[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)

        if verbose:
            print([t-timestep_last[env_id] for env_id in range(eval_batch_size)])
            tensor_debug = array[-1, -context_len:].to(device)
            print('tensor ifelse: ', tensor[-1])
            print('tensor debug: ', tensor_debug)
            print(tensor.shape)        
        
        return tensor

    eval_batch_size = env.num_envs  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_succ = []
    total_overspeed = 0
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
        rtg_pred = rtg_target / rtg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        ctg_pred = ctg_target / ctg_scale * torch.ones((eval_batch_size, 1), dtype=torch.float32, device=device)
        
        while count_done < num_eval_ep: 
            # print(t)
            # zeros place holders

            # init episode
            
            # add (estimated) state in placeholder and normalize
            if use_state_pred:
                img_state = torch.from_numpy(running_state['img']).to(device)
                state_est = model_est.get_state_pred(img_state)
                states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = state_est
            else: 
                states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['state']).to(device)
            # states[:, t] = (states[:, t] - state_mean) / state_std
            image[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['img']).to(device)
            lidar[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['lidar']).to(device)
            
            # calcualate running rtg and add it in placeholder
            if use_value_pred: 
                running_rtg = torch.max(rtg_pred, running_rtg - (running_reward / rtg_scale)).clamp(torch.zeros_like(running_rtg))
                
            else: 
                running_rtg = torch.clamp(running_rtg - (running_reward / rtg_scale), torch.zeros_like(running_rtg))
                        
            rewards_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_rtg.type(torch.FloatTensor))
            # print([t - timestep_last[env_id] for env_id in range(eval_batch_size)])
            
            ts_batch = _get_tensor_batch(timesteps)
            state_batch = _get_tensor_batch(states)
            lidar_batch = _get_tensor_batch(lidar)
            img_batch = _get_tensor_batch(image)
            act_batch = _get_tensor_batch(actions)
            rtg_batch = _get_tensor_batch(rewards_to_go)
            # print(state_batch.shape)
            _, act_preds, rtg_pred = model.forward(ts_batch, state_batch, act_batch, rtg_batch)                        
            act = act_preds[:, -1].detach()
            rtg_pred = rtg_pred[:, -1].detach()
                        
            running_state, running_reward, done, info = env.step(act.cpu().numpy())
            # add action in placeholder
            actions[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = act
            running_cost = np.array([info[idx]['cost'] for idx in range(len(info))])
            total_reward += np.sum(running_reward)
            total_cost += running_cost.sum()
            
            running_reward = CUDA(running_reward).reshape(-1, 1)
            running_cost = CUDA(running_cost).reshape(-1, 1)

            running_overspeed = np.array([info[idx]['velocity_cost']>0. for idx in range(len(info))])
            total_overspeed += np.sum(running_overspeed)
            if render:
                env.render()
            # print(done)
            for i in range(len(done)):
                if done[i]: 
                    total_timesteps += info[i]['episode_length']
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
                    timestep_last[i] = int(t+1)
                    pbar.update(1)
                    if count_done >= num_eval_ep: 
                        break
    
            t += 1
    
    pbar.close()
    results['eval/avg_reward'] = total_reward / count_done
    results['eval/avg_ep_len'] = total_timesteps / count_done
    results['eval/success_rate'] = np.array(total_succ)[:, 0].mean()
    results['eval/oor_rate'] = np.array(total_succ)[:, 1].mean()
    results['eval/crash_rate'] = np.array(total_succ)[:, 2].mean()
    results['eval/max_step'] = np.array(total_succ)[:, 3].mean()
    results['eval/avg_cost'] = total_cost / count_done
    results['eval/over_speed'] = total_overspeed / total_timesteps
    
    env.reset()
    return results


def evaluate_on_env_structure(model, device, context_len, env, rtg_target, ctg_target, rtg_scale, ctg_scale,
                    num_eval_ep=10, max_test_ep_len=1000,
                    state_mean=None, state_std=None, render=False, use_value_pred=False, 
                    use_state_pred=False, model_est=None):
    
    def _get_tensor_batch(array_in, verbose=0):
        array = array_in.clone().detach()
        array_zeros = torch.zeros_like(array)
        tensor = torch.cat([torch.cat([array_zeros[[env_id], :context_len-(t-timestep_last[env_id])], array[[env_id], :t-timestep_last[env_id]]], dim=1) if t-timestep_last[env_id] < context_len \
                else array[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
        
        if verbose:
            print([t-timestep_last[env_id] for env_id in range(eval_batch_size)])
            tensor_debug = array[-1, -context_len:].to(device)
            print('tensor ifelse: ', tensor[-1])
            print('tensor debug: ', tensor_debug)

            print(tensor.shape)        
        return tensor

    eval_batch_size = env.num_envs  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_overspeed = 0
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
            
            # add (estimated) state in placeholder and normalize
            if use_state_pred:
                img_state = torch.from_numpy(running_state['img']).to(device)
                state_est = model_est.get_state_pred(img_state)
                states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = state_est
            else: 
                states[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['state']).to(device)
            # states[:, t] = (states[:, t] - state_mean) / state_std
            image[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['img']).to(device)
            lidar[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = torch.from_numpy(running_state['lidar']).to(device)
            
            # calcualate running rtg and add it in placeholder
            if use_value_pred: 
                running_rtg = torch.max(rtg_pred, running_rtg - (running_reward / rtg_scale)).clamp(torch.zeros_like(running_rtg))
                running_ctg = torch.min(ctg_pred, running_ctg - (running_cost / ctg_scale)).clamp(torch.zeros_like(running_ctg))
                
            else: 
                running_rtg = torch.clamp(running_rtg - (running_reward / rtg_scale), torch.zeros_like(running_rtg))
                running_ctg = torch.clamp(running_ctg - (running_cost / ctg_scale), torch.zeros_like(running_ctg))
                        
            rewards_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_rtg.type(torch.FloatTensor))
            costs_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_ctg.type(torch.FloatTensor))
            # print([t - timestep_last[env_id] for env_id in range(eval_batch_size)])
            
            ts_batch = _get_tensor_batch(timesteps)
            state_batch = _get_tensor_batch(states)
            lidar_batch = _get_tensor_batch(lidar)
            img_batch = _get_tensor_batch(image)
            act_batch = _get_tensor_batch(actions)
            rtg_batch = _get_tensor_batch(rewards_to_go)
            ctg_batch = _get_tensor_batch(costs_to_go)
            # print(state_batch.shape)
            _, act_preds, rtg_pred, ctg_pred = model.forward(ts_batch, [state_batch, lidar_batch, img_batch], act_batch, rtg_batch, ctg_batch, deterministic=True)                        
            act = act_preds[:, -1].detach()
            rtg_pred = rtg_pred[:, -1].detach()
            ctg_pred = ctg_pred[:, -1].detach()
            
            running_state, running_reward, done, info = env.step(act.cpu().numpy())
            # add action in placeholder
            actions[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = act
            running_cost = np.array([info[idx]['cost'] for idx in range(len(info))])
            running_overspeed = np.array([info[idx]['velocity_cost']>0. for idx in range(len(info))])

            total_reward += np.sum(running_reward)
            total_cost += running_cost.sum()
            total_overspeed += np.sum(running_overspeed)
            
            running_reward = CUDA(running_reward).reshape(-1, 1)
            running_cost = CUDA(running_cost).reshape(-1, 1)
            
            if render:
                env.render()
            # print(done)
            for i in range(len(done)):
                if done[i]: 
                    total_timesteps += info[i]['episode_length']
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
                    timestep_last[i] = int(t+1)
                    pbar.update(1)
                    if count_done >= num_eval_ep: 
                        break
            
            t += 1
    
    pbar.close()
    results['eval/avg_reward'] = total_reward / count_done
    results['eval/avg_ep_len'] = total_timesteps / count_done
    results['eval/success_rate'] = np.array(total_succ)[:, 0].mean()
    results['eval/oor_rate'] = np.array(total_succ)[:, 1].mean()
    results['eval/crash_rate'] = np.array(total_succ)[:, 2].mean()
    results['eval/max_step'] = np.array(total_succ)[:, 3].mean()
    results['eval/avg_cost'] = total_cost / count_done
    results['eval/over_speed'] = total_overspeed / total_timesteps
    
    env.reset()
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
                else timesteps[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            state_batch = torch.cat([states[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else states[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            lidar_batch = torch.cat([lidar[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else lidar[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            img_batch = torch.cat([image[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else image[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            act_batch = torch.cat([actions[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else actions[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            rtg_batch = torch.cat([rewards_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else rewards_to_go[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            ctg_batch = torch.cat([costs_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else costs_to_go[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
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
                    total_timesteps += info[i]['episode_length']

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
                    num_eval_ep=10, max_test_ep_len=1000, use_value_pred=False, 
                    state_mean=None, state_std=None, render=False):
    def _get_tensor_batch(array_in, verbose=0):
        array = array_in.clone().detach()
        tensor = torch.cat([array[[env_id], :context_len] if t-timestep_last[env_id] < context_len \
                else array[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)

        if verbose:
            print([t-timestep_last[env_id] for env_id in range(eval_batch_size)])
            tensor_debug = array[-1, -context_len:].to(device)
            print('tensor ifelse: ', tensor[-1])
            print('tensor debug: ', tensor_debug)

            print(tensor.shape)        
        return tensor

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
                running_rtg = torch.max(rtg_pred, running_rtg - (running_reward / rtg_scale)).clamp(torch.zeros_like(running_rtg))
                running_ctg = torch.min(ctg_pred, running_ctg - (running_cost / ctg_scale)).clamp(torch.zeros_like(running_ctg))
                
            else: 
                running_rtg = torch.clamp(running_rtg - (running_reward / rtg_scale), torch.zeros_like(running_rtg))
                running_ctg = torch.clamp(running_ctg - (running_cost / ctg_scale), torch.zeros_like(running_ctg))
            
            rewards_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_rtg.type(torch.FloatTensor))
            costs_to_go[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = CUDA(running_ctg.type(torch.FloatTensor))
            # print([t - timestep_last[env_id] for env_id in range(eval_batch_size)])
            
            ts_batch = _get_tensor_batch(timesteps)
            state_batch = _get_tensor_batch(states)
            lidar_batch = _get_tensor_batch(lidar)
            img_batch = _get_tensor_batch(image)
            act_batch = _get_tensor_batch(actions)
            rtg_batch = _get_tensor_batch(rewards_to_go)
            ctg_batch = _get_tensor_batch(costs_to_go)
            
            # print(state_batch.shape)
            _, act_preds, rtg_pred, ctg_pred = model.forward(ts_batch, [state_batch, lidar_batch, img_batch], act_batch, rtg_batch, ctg_batch, deterministic=True)                        
            act = act_preds[:, -1].detach()
            rtg_pred = rtg_pred[:, -1].detach()
            ctg_pred = ctg_pred[:, -1].detach()

            
            running_state, running_reward, done, info = env.step(act[0].cpu().numpy())
            # add action in placeholder
            actions[range(eval_batch_size), [t - timestep_last[env_id] for env_id in range(eval_batch_size)]] = act
            running_cost = np.array([info['cost']]) # np.array([info[idx]['cost'] for idx in range(len(info))])
            total_reward += running_reward # np.sum(running_reward)
            total_cost += running_cost.sum() # running_cost.sum()
            
            running_reward = CUDA(np.array([running_reward])).reshape(-1, 1)
            running_cost = CUDA(running_cost).reshape(-1, 1)
            
            if render:
                env.render() # mode='top_down', film_size=(800, 800))
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
                else timesteps[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            state_batch = torch.cat([states[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else states[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            lidar_batch = torch.cat([lidar[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else lidar[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            img_batch = torch.cat([image[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else image[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            act_batch = torch.cat([actions[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else actions[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            rtg_batch = torch.cat([rewards_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else rewards_to_go[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
                for env_id in range(eval_batch_size)], dim=0).to(device)
            ctg_batch = torch.cat([costs_to_go[[env_id], :context_len] if t - timestep_last[env_id] < context_len \
                else costs_to_go[[env_id], t-timestep_last[env_id]-context_len:t-timestep_last[env_id]] \
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
