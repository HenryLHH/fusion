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

import random
from metadrive.constants import HELP_MESSAGE

import matplotlib.pyplot as plt

from envs import State_TopDownMetaDriveEnv
from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.component.blocks.first_block import FirstPGBlock

import numpy as np
import time as time
from tqdm import trange
import pickle

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

if __name__ == "__main__":
    folder_name = 'data_bisim_cost_continuous/'
    
    print(HELP_MESSAGE)
    
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
            map=3,
            traffic_density=0.2,
            environment_num=1000,
            accident_prob=0.0,
            start_seed=0, #random.randint(0, 1000),
            distance=20,
            generalized_blocks=[],
            IDM_agent=True, 
            horizon=1000,
        )
    )
    
    # t0 = time.time()
    print(env.config)
    i_last = 0
    
    target = Target(0.375, 30)
    o = env.reset()
    steering_controller = PIDController(1.6, 0.0008, 27.3)
    acc_controller = PIDController(0.1, 0.001, 0.3)
    
    steering_error = - target.lateral
    steering = steering_controller.get_result(steering_error)
    acc_error = env.vehicles[env.DEFAULT_AGENT].speed - target.speed
    acc = acc_controller.get_result(acc_error)
    
    saved_obs = []
    saved_label = []
    saved_act = []
    index_save = 0
    i_last = 1
    
    for i in trange(1, 500001):
        o, r, d, info = env.step([-steering, acc])
        # env.render(mode="top_down", film_size=(800, 800))
        saved_obs.append(o['img'])
        saved_label.append(info)
        # saved_act.append(np.array([-steering, acc]))
        
        o = o['state']
        steering_error = o[0] - target.lateral
        steering = steering_controller.get_result(steering_error)

        t_speed = target.speed if abs(o[12] - 0.5) < 0.01 else target.speed - 10
        acc_error = env.vehicles[env.DEFAULT_AGENT].speed - t_speed
        acc = acc_controller.get_result(acc_error)
        
        # if i % 100 == 0 or info['crash_vehicle']:
        #     # print("Close the popup window to continue.")
        #     draw_multi_channels_top_down_observation(o_vis)
        #     print(info['crash_vehicle'])
        
        # print(info['cost'], info['cost_sparse'])
        if d:
            # print(i-i_last)
            env.reset()
            # t1 = time.time()
            # print('Episode Avg. Step Time: ', (t1 - t0) / (i - i_last))
            # t0 = t1
            i_last = i
            steering_controller.reset()
            steering_error = - target.lateral
            steering = steering_controller.get_result(steering_error)
            
            acc_controller.reset()
            acc_error = env.vehicles[env.DEFAULT_AGENT].speed - target.speed
            acc = acc_controller.get_result(acc_error)

            with open(folder_name+'label/'+str(index_save)+'.pkl', 'wb') as f:
                pickle.dump(saved_label, f)
            np.save(folder_name+'data/'+str(index_save)+'.npy', np.array(saved_obs))
            # np.save(folder_name+'data/'+str(index_save)+'_act.npy', np.array(saved_act))
            
            saved_label, saved_obs = [], []
            index_save += 1
            print('saved episode: ', index_save)