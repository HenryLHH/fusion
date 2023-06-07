
import numpy as np
import torch
from tqdm import trange, tqdm
from utils.utils import CUDA

def evaluate_on_env(model, device, env, num_eval_ep=10, render=False, max_test_ep_len=1000, image=True):

    eval_batch_size = env.num_envs  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_succ = []
    
    state_dim = 35 # env.observation_space.shape[0]
    act_dim = 2 # env.action_space.shape[0]
    model.eval()
    count_done = 0
    pbar = tqdm(total=num_eval_ep)
    timestep_last = np.zeros(eval_batch_size, dtype=np.int32)
    t = 0
    with torch.no_grad():

        running_state = env.reset()
        while count_done < num_eval_ep: 
            total_timesteps += eval_batch_size
            
            img_state = CUDA(running_state['img'])
            lidar_state = CUDA(running_state['lidar'])

            state = CUDA(running_state['state'])
            # print(img_state.shape)
            if image: 
                act_logits = model.act(img_state, lidar_state)               
            else: 
                act_logits = model.act(state, lidar_state)               
            
            act = act_logits[:, :2].detach()
            
            running_state, running_reward, done, info = env.step(act.cpu().numpy())
            # add action in placeholder
            running_cost = np.array([info[idx]['cost_sparse'] for idx in range(len(info))])
            total_reward += np.sum(running_reward)
            total_cost += running_cost.sum()
            if render:
                env.render(mode='top_down', film_size=(800, 800))

            for i in range(len(done)):
                if done[i]: 
                    total_succ.append([info[i]['arrive_dest'], info[i]['out_of_road'], info[i]['crash'], info[i]['max_step']])
                    count_done += 1
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

    return results


def render_env_icil(model, device, env, num_eval_ep=10, render=False, max_test_ep_len=1000):

    eval_batch_size =1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0
    total_cost = 0
    total_succ = []
    
    state_dim = 35 # env.observation_space.shape[0]
    act_dim = 2 # env.action_space.shape[0]
    model.eval()
    count_done = 0
    pbar = tqdm(total=num_eval_ep)
    timestep_last = np.zeros(eval_batch_size, dtype=np.int32)
    t = 0
    with torch.no_grad():

        running_state = env.reset()
        while count_done < num_eval_ep: 
            total_timesteps += eval_batch_size
            
            img_state = CUDA(running_state['img'])
            # print(img_state.shape)
            act_logits = model.policy_network(model.causal_feature_encoder(img_state))                
            act = act_logits[:, :2].detach()
            
            running_state, running_reward, done, info = env.step(act[0].cpu().numpy())
            # add action in placeholder
            running_cost = np.array([info['cost_sparse']]) # np.array([info[idx]['cost_sparse'] for idx in range(len(info))])
            total_reward += running_reward # np.sum(running_reward)
            total_cost += running_cost.sum() # running_cost.sum()
            if render:
                env.render(mode='top_down', film_size=(800, 800))

            if done: 
                total_succ.append([info['arrive_dest'], info['out_of_road'], info['crash'], info['max_step']])
                count_done += 1
                pbar.update(1)
                if count_done >= num_eval_ep: 
                    break
                env.reset()
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
