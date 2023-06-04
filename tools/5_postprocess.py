from itertools import count
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse
import glob

def _compute_cumulative_returns(reward): 
    returns = [0]
    gamma = 0.99
    cumulative_returns = 0
    for r in reversed(reward): 
        cumulative_returns = r + gamma * cumulative_returns
        returns.append(cumulative_returns)
    returns = np.array([i for i in reversed(returns)])
    returns = returns[1:]

    return returns

    

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="data_bisim")
args = parser.parse_args()
os.makedirs(args.filename+'_post', exist_ok=True)
os.makedirs(os.path.join(args.filename+'_post', 'data'), exist_ok=True)
os.makedirs(os.path.join(args.filename+'_post', 'label'), exist_ok=True)

num_files = len(glob.glob(os.path.join(args.filename, 'label/', '*.pkl')))
print('num files: ', num_files)
ts_per_file = 10000
filename_label = [os.path.join(args.filename, 'label/')+str(i)+'.pkl' for i in range(num_files)]
filename_data = [os.path.join(args.filename, 'data/')+str(i)+'.npy' for i in range(num_files)]
# filename_act = ['data_bisim/data/'+str(i)+'_act.npy' for i in range(num_files)]


str_2_id = {'>': 1, 'S': 1, 'r': 1, 'R': 1, 'C': 2, 'X': 3, 'T': 4, 'O': 5}

count_crash, count_oor = np.zeros((6, )), np.zeros((6, ))
count_road_type = np.zeros((6, ))
count_lane_pos = np.zeros((4, ))

pbar = tqdm(total = num_files)

ep_end_id = []

count_trans = 0
ep_rew_list, ep_cost_list = [], []
ep_success_list, ep_crash_list = [], []
for f_data, f_label in zip(filename_data, filename_label):
    pbar.update(1)
    label_dict = pickle.load(open(f_label, 'rb'))
    observations = np.load(f_data, allow_pickle=True)[:-1] # cut the final state
    # print('ep len: ', len(observations))
    ep_rew, ep_cost = 0., 0. 
    num_crash = 0
    last_dist = 1.
    reward = np.array([label_dict[i]['step_reward'] for i in range(len(observations))])
    cost = np.array([label_dict[i]['cost'] for i in range(len(observations))])
    cost_continuous = np.array([max(0, 0.5-np.min(label_dict[i]['lidar_state']))**2 for i in range(len(observations))])
    
    v_r_target = _compute_cumulative_returns(reward)
    v_c_target = _compute_cumulative_returns(cost)
    v_c_target_cont = _compute_cumulative_returns(cost_continuous)

    
    # print(v_r_target.shape)
    # print(v_c_target.shape)
    # print(reward.shape)
    # print(v_c_target)
    # if label_dict[-1]['max_step']: 
    #     count_trans += 1
    
    obs_last = None
    label_true_state_last = None
    label_lidar_state_last = None
    
    for i in range(len(observations)):
        ep_rew += label_dict[i]['step_reward']
        lidar_dist = np.min(label_dict[i]['lidar_state'])
        ep_cost += max(0, np.exp(max(0, 0.4-lidar_dist)) - np.exp(max(0, 0.4-last_dist)))
        last_dist = lidar_dist
        num_crash += label_dict[i]['cost_sparse']

        if obs_last is None: # first action
            obs_last = observations[i]
            label_true_state_last = label_dict[i]['true_state']
            label_lidar_state_last = label_dict[i]['lidar_state']

            continue
        
        # label_crash = (label_dict[i]['crash_object'] or label_dict[i]['crash_vehicle'])
        # label_out_of_road = label_dict[i]['out_of_road']
        # label_lane = label_dict[i]['current_lane']
        label_true_state = label_dict[i]['true_state']
        label_lidar_state = label_dict[i]['lidar_state']
        # label_cost = label_dict[i]['cost']

        action = np.array(label_dict[i]['raw_action'])
        
        obs = observations[i]

        # print(label_dict[i])
        # input()
        
        transitions = [obs_last, action, obs]
        lidar_transitions = [label_lidar_state_last, action, label_lidar_state]
        label_transitions = [label_true_state_last, action, label_true_state]

        label_cur = label_dict[i]
        label_cur.pop('lidar_state')
        label_cur.pop('true_state')
        # print(label_cur.keys())
        
        label_cur.update({'value_reward_target': v_r_target[i], 
                          'value_cost_target': v_c_target[i],
                          'value_cost_target_cont': v_c_target_cont[i]})
        
        np.save(os.path.join(args.filename+'_post', 'label/')+str(count_trans)+'.npy', label_cur)
        np.save(os.path.join(args.filename+'_post', 'data/')+str(count_trans)+'.npy', [transitions, lidar_transitions, label_transitions])
        count_trans += 1
        
        # get stats
        # count_crash[label_road_type] += label_crash
        # count_oor[label_road_type] += label_out_of_road
        # count_road_type[label_road_type] += 1
        # count_lane_pos[label_lane_pos] += 1
        
        # if label_dict[i]['crash_vehicle'] or label_dict[i]['crash_object'] or label_dict[i]['out_of_road'] or label_dict[i]['arrive_dest'] or label_dict[i]['crash'] or label_dict[i]['max_step']:        
        #     obs_last = None
        #     label_true_state = None
        #     # print('Count Transitions: ', count_trans, i)
        #     ep_end_id.append(count_trans)
        #     continue

        obs_last = obs
        label_true_state_last = label_true_state
        
    
    # print('ep_rew: ', ep_rew/len(observations), 'ep_cost: ', ep_cost/len(observations))
    # print('ep_crash: ', num_crash, 'ep_succeed: ', label_dict[i]['arrive_dest'] )
    ep_rew_list.append(ep_rew/len(observations))
    ep_cost_list.append(ep_cost)
    ep_crash_list.append(num_crash)
    ep_success_list.append(label_dict[i]['arrive_dest'])
print(count_trans)
ep_success = np.array(ep_success_list)
ep_crash = np.array(ep_crash_list)
ep_rew_arr = np.array(ep_rew_list)
ep_cost_arr = np.array(ep_cost_list)

from matplotlib import pyplot as plt
plt.figure()
plt.scatter(ep_cost_arr, ep_crash)
plt.savefig('cost_arr')
plt.close()
plt.figure()
plt.scatter(ep_cost_arr, ep_rew_arr)
plt.savefig('cost_reward')
plt.close()
success_idx = np.where(ep_success)[0]
failure_idx = np.where(ep_success==False)[0]
print('success: ', len(success_idx), 'failure: ', len(failure_idx))
print('success: ', np.mean(ep_rew_arr[success_idx]), np.std(ep_rew_arr[success_idx]))
print('failure: ', np.mean(ep_rew_arr[failure_idx]), np.std(ep_rew_arr[failure_idx]))
print('success: ', np.mean(ep_cost_arr[success_idx]), np.std(ep_cost_arr[success_idx]))
print('failure: ', np.mean(ep_cost_arr[failure_idx]), np.std(ep_cost_arr[failure_idx]))
print('success: ', np.mean(ep_crash[success_idx]), np.std(ep_crash[success_idx]))
print('failure: ', np.mean(ep_crash[failure_idx]), np.std(ep_crash[failure_idx]))
# np.save('data_bisim_vector/ep_end_id.npy', np.array(ep_end_id))

# print('Out of Road: ', count_oor, 'Crash: ', count_crash)
# print('Lane position: ', count_lane_pos, 'Road Type: ', count_road_type)