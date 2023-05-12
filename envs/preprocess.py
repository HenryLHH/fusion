from itertools import count
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="data_bisim")
args = parser.parse_args()
os.makedirs(args.filename+'_post')
os.makedirs(os.path.join(args.filename+'_post', 'data'))
os.makedirs(os.path.join(args.filename+'_post', 'label'))

num_files = 50
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
obs_last = None
label_true_state_last = None
count_trans = 0
for f_data, f_label in zip(filename_data, filename_label):
    pbar.update(1)
    label_dict = pickle.load(open(f_label, 'rb'))
    observations = np.load(f_data)    
    
    for i in range(ts_per_file):
        if obs_last is None: # first action
            obs_last = observations[i]
            label_true_state_last = label_dict[i]['true_state']
            continue
        
        # label_crash = (label_dict[i]['crash_object'] or label_dict[i]['crash_vehicle'])
        # label_out_of_road = label_dict[i]['out_of_road']
        # label_lane = label_dict[i]['current_lane']
        label_true_state = label_dict[i]['true_state']
        # label_cost = label_dict[i]['cost']

        action = np.array(label_dict[i]['raw_action'])
        
        obs = observations[i]

        # print(label_dict[i])
        # input()
        
        transitions = [obs_last, action, obs]
        label_transitions = [label_true_state_last, action, label_true_state]

        
        # if label_lane is not None:
        #     if '>' in label_lane[0]:
        #         label_road_type_prev = 0
        #     else:
        #         label_road_type_prev = str_2_id[label_lane[0].strip('-')[1]]
        #     if '>' in label_lane[1]:
        #         label_road_type = 0
        #     else:
        #         label_road_type = str_2_id[label_lane[1].strip('-')[1]]
            
        #     label_lane_pos = label_lane[2]
        # else: # Anomalies
        #     label_road_type, label_road_type_prev = 0, 0
        # label_cur = np.array([label_crash, label_out_of_road, label_cost, label_road_type, label_road_type_prev, label_lane_pos, label_transitions, label_true_state])
        label_cur = label_dict[i]
        np.save(os.path.join(args.filename+'_post', 'label/')+str(count_trans)+'.npy', label_cur)
        np.save(os.path.join(args.filename+'_post', 'data/')+str(count_trans)+'.npy', [transitions, label_transitions])
        count_trans += 1
        
        #  get stats
        # count_crash[label_road_type] += label_crash
        # count_oor[label_road_type] += label_out_of_road
        # count_road_type[label_road_type] += 1
        # count_lane_pos[label_lane_pos] += 1
        
        if label_dict[i]['crash_vehicle'] or label_dict[i]['crash_object'] or label_dict[i]['out_of_road'] or label_dict[i]['arrive_dest'] or label_dict[i]['crash'] or label_dict[i]['max_step']:        
            obs_last = None
            label_true_state = None
            # print('Count Transitions: ', count_trans, i)
            ep_end_id.append(count_trans)
            continue
        obs_last = obs
        label_true_state_last = label_true_state

# np.save('data_bisim_vector/ep_end_id.npy', np.array(ep_end_id))

# print('Out of Road: ', count_oor, 'Crash: ', count_crash)
# print('Lane position: ', count_lane_pos, 'Road Type: ', count_road_type)
