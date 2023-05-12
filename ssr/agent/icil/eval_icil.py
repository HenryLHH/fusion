
import torch

from tqdm import trange

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