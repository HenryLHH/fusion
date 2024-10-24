import os
import os.path as osp
import random
import uuid
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

from envs.envs import State_TopDownMetaDriveEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.component.pgblock.first_block import FirstPGBlock
from tqdm import tqdm

def make_envs(): 
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=False,
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
        horizon=999,
        # IDM_agent=True,
        # resolution_size=64,
        # generalized_blocks=tune.grid_search([['X', 'T']])
    )
    return State_TopDownMetaDriveEnv(config)


block_list=["S", "T", "R", "X"]

def make_envs_single(block_id=0): 
    idx = int(block_id // 4)
    block_type=block_list[idx]
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=False,
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
        horizon=999,
    )
    return State_TopDownMetaDriveEnv(config)


def make_envs_small_data(): 
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=True,
        random_traffic=False,
        accident_prob=0,
        vehicle_config={"vehicle_model": "default",
            "spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 1),
            "lidar": dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
        },
        map_config={
                "type": "block_sequence", 
                "config": "X" 
        },
        traffic_density=0.2, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Hybrid,
        horizon=999,
        # IDM_agent=True,
        # resolution_size=64,
        # generalized_blocks=tune.grid_search([['X', 'T']])
    )
    return State_TopDownMetaDriveEnv(config)


def seed_all(seed=1029, others: Optional[list] = None) -> None:
    """Fix the seeds of `random`, `numpy`, `torch` and the input `others` object.

    :param int seed: defaults to 1029
    :param Optional[list] others: other objects that want to be seeded, defaults to None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if others is not None:
        if hasattr(others, "seed"):
            others.seed(seed)
            return True
        try:
            for item in others:
                if hasattr(item, "seed"):
                    item.seed(seed)
        except:
            pass


def get_cfg_value(config, key):
    if key in config:
        value = config[key]
        if isinstance(value, list):
            suffix = ""
            for i in value:
                suffix += str(i)
            return suffix
        return str(value)
    for k in config.keys():
        if isinstance(config[k], dict):
            res = get_cfg_value(config[k], key)
            if res is not None:
                return res
    return "None"


def load_config_and_model(path: str, best: bool = False):
    """
    Load the configuration and trained model from a specified directory.

    :param path: the directory path where the configuration and trained model are stored.
    :param best: whether to load the best-performing model or the most recent one.
        Defaults to False.

    :return: a tuple containing the configuration dictionary and the trained model.
    :raises ValueError: if the specified directory does not exist.
    """
    if osp.exists(path):
        config_file = osp.join(path, "config.yaml")
        print(f"load config from {config_file}")
        with open(config_file) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        model_file = "model.pt"
        if best:
            model_file = "model_best.pt"
        model_path = osp.join(path, "checkpoint/" + model_file)
        print(f"load model from {model_path}")
        model = torch.load(model_path)
        return config, model
    else:
        raise ValueError(f"{path} doesn't exist!")


# naming utils


def to_string(values):
    """
    Recursively convert a sequence or dictionary of values to a string representation.

    :param values: the sequence or dictionary of values to be converted to a string.
    :return: a string representation of the input values.
    """
    name = ""
    if isinstance(values, Sequence) and not isinstance(values, str):
        for i, v in enumerate(values):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(v)
        return name
    elif isinstance(values, Dict):
        for i, k in enumerate(sorted(values.keys())):
            prefix = "" if i == 0 else "_"
            name += prefix + to_string(values[k])
        return name
    else:
        return str(values)


DEFAULT_SKIP_KEY = [
    "task", "reward_threshold", "logdir", "worker", "project", "group", "name", "prefix",
    "suffix", "save_interval", "render", "verbose", "save_ckpt", "training_num",
    "testing_num", "epoch", "device", "thread"
]

DEFAULT_KEY_ABBRE = {
    "cost_limit": "cost",
    "mstep_iter_num": "mnum",
    "estep_iter_num": "enum",
    "estep_kl": "ekl",
    "mstep_kl_mu": "kl_mu",
    "mstep_kl_std": "kl_std",
    "mstep_dual_lr": "mlr",
    "estep_dual_lr": "elr",
    "update_per_step": "update"
}


def auto_name(
    default_cfg: dict,
    current_cfg: dict,
    prefix: str = "",
    suffix: str = "",
    skip_keys: list = DEFAULT_SKIP_KEY,
    key_abbre: dict = DEFAULT_KEY_ABBRE
) -> str:
    """Automatic generate the name by comparing the current config with the default one.

    :param dict default_cfg: a dictionary containing the default configuration values.
    :param dict current_cfg: a dictionary containing the current configuration values.
    :param str prefix: (optional) a string to be added at the beginning of the generated
        name.
    :param str suffix: (optional) a string to be added at the end of the generated name.
    :param list skip_keys: (optional) a list of keys to be skipped when generating the
        name.
    :param dict key_abbre: (optional) a dictionary containing abbreviations for keys in
        the generated name.

    :return str: a string representing the generated experiment name.
    """
    name = prefix
    for i, k in enumerate(sorted(default_cfg.keys())):
        if default_cfg[k] == current_cfg[k] or k in skip_keys:
            continue
        prefix = "_" if len(name) else ""
        value = to_string(current_cfg[k])
        # replace the name with abbreviation if key has abbreviation in key_abbre
        if k in key_abbre:
            k = key_abbre[k]
        # Add the key-value pair to the name variable with the prefix
        name += prefix + k + value
    if len(suffix):
        name = name + "_" + suffix if len(name) else suffix

    name = "default" if not len(name) else name
    name = f"{name}-{str(uuid.uuid4())[:4]}"
    return name


@torch.no_grad()
def evaluate_rollouts(model, env, num_eval_ep=50):
    """
    Evaluates the performance of the model on a single episode.
    """
    obs = env.reset()
    n_envs = env.num_envs
    results = {}
    episode_rets, episode_costs, episode_lens = [], [], []
    episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
    count_ep = 0
    success = []
    total_reward = 0
    total_cost = 0
    total_overspeed = 0
    total_timesteps = 0
    pbar = tqdm(total=num_eval_ep)
    while count_ep < num_eval_ep: 
        act, _ =  model.act(obs['state'], True, True, batch_inputs=True)
        obs_next, reward, done, info = env.step(act)
        cost = np.array([info[i]["cost"] for i in range(n_envs)])
        obs = obs_next
        episode_ret += reward.sum()
        episode_len += n_envs
        episode_cost += cost.sum()
        running_overspeed = np.array([info[idx]['velocity_cost']>0. for idx in range(len(info))])
        total_overspeed += np.sum(running_overspeed)
        for i in range(n_envs):
            if done[i]:
                count_ep += 1
                total_reward += info[i]['episode_reward']
                total_timesteps += info[i]['episode_length']
                success.append([info[i]['arrive_dest'], info[i]['out_of_road'], info[i]['crash'], info[i]['max_step']])
                pbar.update(1)
                if count_ep >= num_eval_ep: 
                    break
    pbar.close()
    results['eval/avg_reward'] = total_reward / count_ep
    results['eval/avg_ep_len'] = total_timesteps / count_ep
    results['eval/success_rate'] = np.array(success)[:, 0].mean()
    results['eval/oor_rate'] = np.array(success)[:, 1].mean()
    results['eval/crash_rate'] = np.array(success)[:, 2].mean()
    results['eval/max_step'] = np.array(success)[:, 3].mean()
    results['eval/avg_cost'] = episode_cost / count_ep
    results['eval/over_speed'] = total_overspeed / total_timesteps
    
    return results
