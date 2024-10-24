import os
import numpy as np
import torch
import argparse

from stable_baselines3.common.vec_env import SubprocVecEnv
from metadrive.manager.traffic_manager import TrafficMode
from envs.envs import State_TopDownMetaDriveEnv

from fusion.agent.DT.utils import evaluate_on_env_structure, evaluate_on_env_structure_pred
from fusion.agent.DT.model import SafeDecisionTransformer_Structure
from fusion.agent.icil.eval_icil import evaluate_on_env
from fusion.agent.icil.icil import ICIL
from fusion.agent.bisim.eval_cnn import evaluate_on_env_cnn
from fusion.encoder.model_actor import BisimEncoder_Head_BP_Actor
from fusion.encoder.model_state_est import BisimEncoder_Head_BP_Actor as StatePred


from utils.utils import CUDA

def make_envs(): 
    config = dict(
        environment_num=args.env_num, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=args.seed, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=True,
        random_traffic=False,
        accident_prob=0,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config="TRO"), 
        traffic_density=0.2, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Trigger,
        horizon=args.horizon-1,
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

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="ssr", help="methods to evaluate")
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
    parser.add_argument("--seed", type=int, default=0, help="checkpoint to load")
    parser.add_argument("--env_num", type=int, default=50, help="checkpoint to load")
    parser.add_argument("--hidden", type=int, default=64, help='hidden dim of DT')
    parser.add_argument("--context", type=int, default=30, help='context len of DT')
    parser.add_argument("--horizon", type=int, default=1000, help='horizon of a task')

    parser.add_argument("--cost", type=float, default=0.0, help='cost tolerance of DT')
    parser.add_argument("--dynamics", action="store_true", help='train world dynamics')
    parser.add_argument("--value", action="store_true", help='train value model')
    parser.add_argument("--single_env", action="store_true", help='use single block envs')
    

    return parser


if __name__ == '__main__':
    args = get_train_parser().parse_args()
    if args.single_env: 
        env = SubprocVecEnv([lambda: make_envs_single(i) for i in range(16)], start_method="spawn")    
    else: 
        env = SubprocVecEnv([make_envs for _ in range(16)], start_method="spawn")
    
    if args.method == 'ssr': 
        model = CUDA(SafeDecisionTransformer_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=args.hidden, context_len=args.context, n_heads=1, drop_p=0.1, max_timestep=args.horizon))
        model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'best.pt')))
        print('model loaded')
        results = evaluate_on_env_structure(model, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=350, ctg_target=args.cost, 
                                                    rtg_scale=300.0, ctg_scale=80.0, num_eval_ep=50, max_test_ep_len=args.horizon, use_value_pred=args.value)
    
    elif args.method == 'ssr_pred': 
        model = CUDA(SafeDecisionTransformer_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=args.hidden, context_len=args.context, n_heads=1, drop_p=0.1, max_timestep=args.horizon))
        model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'best.pt')))
        model_est = CUDA(StatePred(hidden_dim=64, output_dim=35, causal=True))
        model_est.load_state_dict(torch.load('checkpoint/state_est.pt'))
        
        print('model loaded')
        results = evaluate_on_env_structure_pred(model, model_est, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=350, ctg_target=0., 
                                                    rtg_scale=40.0, ctg_scale=10.0, num_eval_ep=50, max_test_ep_len=args.horizon)

    elif args.method == 'icil': 
        model = CUDA(ICIL(state_dim=5, action_dim=2, hidden_dim_input=64, hidden_dim=64))
        model.load_state_dict(torch.load('checkpoint/icil/'+args.model+'.pt'))
        results = evaluate_on_env(model, torch.device('cuda:0'), env, num_eval_ep=50)
    elif args.method == 'cnn': 
        model = CUDA(BisimEncoder_Head_BP_Actor(hidden_dim=64, output_dim=2, causal=True))
        model.load_state_dict(torch.load('checkpoint/'+args.model+'.pt'))
        results = evaluate_on_env_cnn(model, torch.device('cuda:0'), env, num_eval_ep=50)
        
    else: 
        raise NotImplementedError
    
    eval_avg_reward = results['eval/avg_reward']
    eval_avg_ep_len = results['eval/avg_ep_len']
    eval_avg_succ = results['eval/success_rate']
    eval_avg_crash = results['eval/crash_rate']
    eval_avg_oor = results['eval/oor_rate']
    eval_avg_max_step = results['eval/max_step']
    eval_avg_cost = results['eval/avg_cost']

    log_str = ("=" * 60 + '\n' +
        "eval avg reward: " + format(eval_avg_reward, ".5f") + '\n' + 
        "eval avg ep len: " + format(eval_avg_ep_len, ".5f") + '\n' +
        "eval avg succ: " + format(eval_avg_succ, ".5f") + '\n' +
        "eval avg crash: " + format(eval_avg_crash, ".5f") + '\n' +
        "eval avg oor: " + format(eval_avg_oor, ".5f") + '\n' +
        "eval avg overtime: " + format(eval_avg_max_step, ".5f") + '\n' +
        "eval avg cost: " + format(eval_avg_cost, ".5f") + '\n'
        )
    print(log_str)
    env.close()