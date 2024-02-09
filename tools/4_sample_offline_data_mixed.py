"""
This file illustrate how to use top-down renderer to provide observation in form of multiple channels of semantic maps.

We let the target vehicle moving forward directly. You can also try to control the vehicle by yourself. See the config
below for more information.

This script will popup a Pygame window, but that is not the form of the observation. We will also popup a matplotlib
window, which shows the details observation of the top-down pygame renderer.

The detailed implementation of the Pygame renderer is in TopDownMultiChannel Class (a subclass of Observation Class)
at: metadrive/obs/top_down_obs_multi_channel.py

We welcome contributions to propose a better representation of the top-down semantic observation!
"""

import os
import random
import argparse
import numpy as np
import time as time
from tqdm import trange
import pickle
import matplotlib.pyplot as plt


from metadrive.constants import HELP_MESSAGE
from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.component.pgblock.first_block import FirstPGBlock

from envs.envs import State_TopDownMetaDriveEnv
from fusion.agent.expert.idm_custom import IDMPolicy_CustomSpeed

def draw_multi_channels_top_down_observation(obs):
    num_channels = obs.shape[-1]
    assert num_channels == 5
    channel_names = [
        "Road and navigation", "Ego now and previous pos", "Neighbor at step t", "Neighbor at step t-1",
        "Neighbor at step t-2"
    ]
    fig, axs = plt.subplots(1, num_channels, figsize=(15, 4), dpi=80)
    count = 0

    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window

    timer = fig.canvas.new_timer(interval=4500)  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    for i, name in enumerate(channel_names):
        count += 1
        ax = axs[i]
        ax.imshow(obs[..., i], cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        # print("Drawing {}-th semantic map!".format(count))
    fig.suptitle("Multi-channels Top-down Observation")
    timer.start()
    plt.show()

class Target:
    def __init__(self, target_lateral, target_speed):
        self.lateral = target_lateral
        self.speed = target_speed

    def go_right(self):
        self.lateral += 0.25 if self.lateral < 0.625 else 0

    def go_left(self):
        self.lateral -= 0.25 if self.lateral > 0.125 else 0

    def faster(self):
        self.speed += 10
    
    def slower(self):
        self.speed -= 10

def collect_data(env, step=150000): 
    global index_save
    target = Target(0.375, 30)
    o = env.reset()
    
    saved_obs = []
    saved_label = []
    saved_act = []

    cost_list, rew_list = [], []
    
    r_last, c_last = 0, 0
    
    for i in trange(1, step+1):
        saved_obs.append(o["img"])
        o, r, d, info = env.step([0., 0.])
        # env.render(mode="top_down", film_size=(800, 800))
        saved_label.append(info)

        # saved_act.append(np.array([-steering, acc]))
        r_last += r
        c_last += info["cost"]

        if d:
            # record the terminal state
            saved_obs.append(o["img"])
            # print(i-i_last)
            # print("r | c: ", r_last, c_last)
            # print("success", info["arrive_dest"])

            # print('Episode length: ', len(saved_label))
            if info["arrive_dest"] or info["max_step"]: 
                with open(os.path.join("dataset", folder_name, "label", str(index_save)+".pkl"), "wb") as f:
                    pickle.dump(saved_label, f)
                np.save(os.path.join("dataset", folder_name, "data", str(index_save)+".npy"), np.array(saved_obs))            
                index_save += 1
                cost_list.append(c_last)
                rew_list.append(r_last)
            else: 
                print("Unsuccessful")
            saved_label, saved_obs = [], []
            o = env.reset()
            r_last = 0
            c_last = 0
            # print("saved episode: ", index_save)
    
    return cost_list, rew_list

def sample_in_envs(block_list=["S", "O", "R", "T"], density=0.2, target_speed=30): 
    cost_list_plot, rew_list_plot = [], []
    for block in block_list: 
        print("========================")
        print(block, target_speed)
            
        env = State_TopDownMetaDriveEnv(
            dict(
                # We also support using two renderer (Panda3D renderer and Pygame renderer) simultaneously. You can
                # try this by uncommenting next line.
                use_render=False,

                # You can also try to uncomment next line with "use_render=True", so that you can control the ego vehicle
                # with keyboard in the main window.
                # manual_control=True,
                vehicle_config={"vehicle_model": "default",
                    "spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 1),
                    "lidar": dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
                },
                safe_rl_env=True,
                traffic_mode=TrafficMode.Hybrid, 
                # map=1,
                map_config={
                    "type": "block_sequence", 
                    "config": block 
                },
                traffic_density=density,
                environment_num=10,
                accident_prob=0.0,
                start_seed=0, #random.randint(0, 1000),
                distance=20,
                random_traffic=True,
                # generalized_blocks=["X", "S", "C"],
                agent_policy=IDMPolicy_CustomSpeed, 
                idm_target_speed=target_speed,
                # idm_acc_factor=target_speed/10., 
                horizon=999,
            )
        )
        # t0 = time.time()
        # print(env.config)
        cost_list, rew_list = collect_data(env, args.timestep)
        rew_list_plot += rew_list
        cost_list_plot += cost_list
        env.close()
    
    return cost_list_plot, rew_list_plot


def get_parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_default", help="dataset name")
    parser.add_argument("--timestep", type=int, default=100_000, help="number of episodes")

    return parser
if __name__ == "__main__":
    args = get_parser().parse_args()
    folder_name = args.dataset # "data_mixed_dynamics/"
    os.makedirs("dataset", exist_ok=True)
    os.makedirs(os.path.join("dataset", args.dataset), exist_ok=True)
    os.makedirs(os.path.join("dataset", args.dataset, "data"), exist_ok=True)
    os.makedirs(os.path.join("dataset", args.dataset, "label"), exist_ok=True)
    
    print(HELP_MESSAGE)
    
    global index_save
    index_save = 0
    plt.figure()

    cost_list, rew_list = sample_in_envs(["S", "O", "R", "T"], 0.2, target_speed=10)
    plt.scatter(cost_list, rew_list)
    print("10: ", np.mean(cost_list), np.mean(rew_list))
    
    cost_list, rew_list = sample_in_envs(["S", "O", "R", "T"], 0.2, target_speed=20)
    plt.scatter(cost_list, rew_list)
    print("30: ", np.mean(cost_list), np.mean(rew_list))
    
    cost_list, rew_list = sample_in_envs(["S", "O", "R", "T"], 0.2, target_speed=30)
    plt.scatter(cost_list, rew_list)
    print("50: ", np.mean(cost_list), np.mean(rew_list))
    
    cost_list, rew_list = sample_in_envs(["S", "O", "R", "T"], 0.2, target_speed=40)
    plt.scatter(cost_list, rew_list)
    print("70: ", np.mean(cost_list), np.mean(rew_list))
    
    plt.legend(['speed={:2d}'.format(i) for i in [10, 20, 30, 40]])
    plt.savefig('cr_plot_offline')

    # cost_list, rew_list = sample_in_envs(["X", "C", "r"], 0.0)