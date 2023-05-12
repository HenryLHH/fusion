
from audioop import avg
from gc import callbacks
from math import gamma
import numpy as np
import argparse
from pathlib import Path

import copy
import argparse
import random
from tqdm import tqdm
import torch.multiprocessing as mp
import torch
import torch.nn as nn
from metadrive import MetaDriveEnv, SafeMetaDriveEnv, TopDownMetaDrive
from metadrive.constants import HELP_MESSAGE
from metadrive.component.map.base_map import BaseMap, MapGenerateMethod, parse_map_config
from metadrive.component.blocks.first_block import FirstPGBlock
from metadrive.manager.traffic_manager import TrafficMode

# from agents import DQN, MBRL, DQN_HER
from stable_baselines3 import A2C, PPO, SAC, DDPG #, MBRL
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from bc_model import BC_Agent
from envs import State_TopDownMetaDriveEnv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', help='train or test')


args = parser.parse_args()

metadrive_test_config = dict(
    use_render=False,
    manual_control=False,
    traffic_density=0.3,
    environment_num=10,
    random_agent_model=False,
    random_lane_width=False,
    random_lane_num=False,
    traffic_mode=TrafficMode.Hybrid,
    map=3, #"CSR", #'SOCrO', #'SCrROYS',  # seven block
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM, #BIG_BLOCK_NUM, #BIG_BLOCK_SEQUENCE, #BIG_BLOCK_NUM, # 
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,#3.5,
        BaseMap.LANE_NUM: 3, #3,
        "exit_length": 50,
    },
    safe_rl_env=True,
    start_seed=0, #random.randint(0, 1000)
    # accident_prob=0.8, # 0.8
    vehicle_config = {"vehicle_model": "default",
                    "lidar": {"num_lasers": 1, "distance": 50, "num_others": 4, "gaussian_noise": 0.0, "dropout_prob": 0.0},
                    "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1)},
)


def rollout_eval(test_env, n_envs=100, gpcg_sample=None):
    success_count, oor_count, crash_count, ot_count,  = 0, 0, 0, 0
    success_rate, oor_rate, crash_rate, ot_rate, avg_reward, avg_cost = 0, 0, 0, 0, 0, 0
    reward_count, cost_count,success_count_list = [], [], []
    if n_envs > 0:
        test_env.reset()
        for i in tqdm(range(n_envs)): # and not dones:
            dones = False
            obs = test_env.reset()
            step_count = 0
            while not dones and step_count < 1000:
                step_count += 1
                action, _ = model(torch.from_numpy(obs['img'][None, :, :, :]), torch.from_numpy(obs['state'][None, :]), random=False)
                action = action[0].detach().numpy()
                action = np.clip(action, -np.ones(2, ), np.ones(2))
                obs, _, dones, info = test_env.step(action)
                test_env.render(mode='top_down')
            # dones=False
            success_count_list.append(info["arrive_dest"])
            if info["arrive_dest"] and info["out_of_road"]:
                success_count += 1
            else:
                if info["arrive_dest"]: success_count += 1
                if info["out_of_road"]: oor_count += 1
                if info["crash"]: crash_count += 1
                if info["max_step"]: ot_count += 1
            cost_count.append(info["cost"] / info["episode_length"])
            reward_count.append(info["episode_reward"] / info["episode_length"])

        # print(success_count)
        success_rate = success_count / n_envs
        oor_rate = oor_count / n_envs
        crash_rate = crash_count / n_envs
        ot_rate = ot_count / n_envs
        avg_reward = np.mean(reward_count)
        avg_cost = np.mean(cost_count)
        print("reward: %.4f, cost: %.4f, success rate: %.2f, oor_rate: %.2f, crash_rate: %.2f, ot_rate: %.2f" % (avg_reward, avg_cost, success_rate, oor_rate, crash_rate, ot_rate))

    return avg_reward, avg_cost, reward_count, cost_count, success_rate, oor_rate, crash_rate, ot_rate, success_count_list

if __name__ == '__main__':
    model = BC_Agent()
    model.load_state_dict(torch.load(ROOT/'agent_bc.pt'))
    test_env = State_TopDownMetaDriveEnv(metadrive_test_config)

    rollout_eval(test_env, n_envs=100) 
