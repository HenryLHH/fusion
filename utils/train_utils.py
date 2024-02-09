import argparse
import logging
import os

import torch
import numpy as np

from metadrive.manager.traffic_manager import TrafficMode
from envs.envs import State_TopDownMetaDriveEnv

def CPU(x):
    return x.detach().cpu().numpy()

def CUDA(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.cuda()

def make_envs(args): 
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=True,
        random_traffic=False,
        accident_prob=0,
        distance=20,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config="TRO"), 
        traffic_density=0.2, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Trigger,
        horizon=args.horizon-1,
    )
    return State_TopDownMetaDriveEnv(config)

block_list=["S", "T", "R", "O"]

def make_envs_single(args, block_id=0): 
    idx = int(block_id // 4)
    block_type=block_list[idx]
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=True,
        random_traffic=False,
        accident_prob=0,
        distance=20,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config=block_type), 
        traffic_density=0.2, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Hybrid,
        horizon=args.horizon-1,
    )
    return State_TopDownMetaDriveEnv(config)



def setup_logger(debug=False):
    import logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )
