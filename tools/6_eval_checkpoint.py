import os
import numpy as np
import torch
import argparse

from stable_baselines3.common.vec_env import SubprocVecEnv
from metadrive.manager.traffic_manager import TrafficMode
from envs.envs import State_TopDownMetaDriveEnv

from fusion.agent.DT.utils import evaluate_on_env_structure, evaluate_expert, evaluate_on_env_structure_pred, evaluate_on_env_nocost
from fusion.agent.DT.model import DecisionTransformer, SafeDecisionTransformer_Structure

from fusion.agent.icil.eval_icil import evaluate_on_env as eval_icil
from fusion.agent.icil.icil_state import ICIL

from fusion.agent.bisim.eval_cnn import evaluate_on_env_cnn

from fusion.agent.bnn.bnn import BNN_Agent
from fusion.agent.bnn.eval_utils import evaluate_on_env as eval_bnn

from fusion.agent.gsa.gsa import GSA_Agent
from fusion.agent.gsa.eval_utils import evaluate_on_env as eval_gsa

from fusion.agent.DAgger.bc_model import BC_Agent
from fusion.agent.DAgger.eval_utils import evaluate_on_env as eval_bc

from fusion.agent.bearl.bearl import BEARL
from fusion.configs.bearl_configs import BEARLTrainConfig, BEARL_DEFAULT_CONFIG
from fusion.agent.bcql.bcql import BCQL
from fusion.configs.bcql_configs import BCQL_DEFAULT_CONFIG
from fusion.agent.cpq.cpq import CPQ
from fusion.configs.cpq_configs import CPQ_DEFAULT_CONFIG

from utils.exp_utils import evaluate_rollouts

from fusion.encoder.model_actor import BisimEncoder_Head_BP_Actor
from fusion.encoder.model_state_est import BisimEncoder_Head_BP_Actor as StatePred
from fusion.agent.expert.idm_custom import IDMPolicy_CustomSpeed


from utils.utils import CUDA

def make_envs(generalize=False):
    block_seq = "TRO" if not generalize else "SCS" 
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=args.seed, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=False,
        random_traffic=False,
        accident_prob=0,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config=block_seq), 
        traffic_density=0.2, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Trigger,
        horizon=args.horizon-1,
        # IDM_agent=True,
        # resolution_size=64,
        # generalized_blocks=tune.grid_search([['X', 'T']])
    )
    return State_TopDownMetaDriveEnv(config)


def make_envs_single(block_id=0, generalize=False): 
    block_list=["SSS", "T", "R", "X"] # if not generalize else ["C", "X", "r", "O"]
    # block_list = [args.task]*4
    traffic_density = 0.2 if not generalize else 0.3
    idx = int(block_id // 4)
    block_type=block_list[idx]
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=False,
        random_traffic=True,
        accident_prob=0,
        distance=20,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        map_config=dict(type="block_sequence", config=block_type),  # args.task
        traffic_density=traffic_density, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Hybrid,
        horizon=args.horizon-1,
    )
    return State_TopDownMetaDriveEnv(config)



def make_envs_expert(block_id=0, generalize=False): 
    block_list=["SSS", "T", "R", "X"] if not generalize else ["C", "X", "r", "O"]
    # block_list = [args.task]*4
    traffic_density = 0.3
    idx = int(block_id // 4)
    block_type=block_list[idx]
    config = dict(
        environment_num=10, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
        start_seed=0, #tune.grid_search([0, 1000]),
        frame_stack=3, # TODO: debug
        safe_rl_env=False,
        random_traffic=True,
        accident_prob=0,
        distance=20,
        vehicle_config=dict(lidar=dict(
            num_lasers=240,
            distance=50,
            num_others=4
        )),
        agent_policy=IDMPolicy_CustomSpeed, 
        idm_target_speed=5+10*block_id,
        map_config=dict(type="block_sequence", config=args.task), 
        traffic_density=traffic_density, #tune.grid_search([0.05, 0.2]),
        traffic_mode=TrafficMode.Hybrid,
        horizon=args.horizon-1,
    )
    env = State_TopDownMetaDriveEnv(config)
    return env

def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="ssr", help="methods to evaluate")
    parser.add_argument("--model", type=str, default="encoder", help="checkpoint to load")
    parser.add_argument("--task", type=str, default="O", help="task of road")
    
    parser.add_argument("--seed", type=int, default=0, help="random seed of env")
    parser.add_argument("--num_envs", type=int, default=16, help="number of parallel environments")
    
    parser.add_argument("--ep_num", type=int, default=50, help="episodes to evaluate")
    parser.add_argument("--num_trials", type=int, default=5, help="number of random experiments")

    parser.add_argument("--hidden", type=int, default=64, help='hidden dim of DT')
    parser.add_argument("--context", type=int, default=30, help='context len of DT')
    parser.add_argument("--horizon", type=int, default=1000, help='horizon of a task')

    parser.add_argument("--reward", type=float, default=350.0, help='reward expectation of DT')
    parser.add_argument("--cost", type=float, default=0.0, help='cost tolerance of DT')
    parser.add_argument("--dynamics", action="store_true", help='train world dynamics')
    parser.add_argument("--value", action="store_true", help='train value model')
    parser.add_argument("--single_env", action="store_true", help='use single block envs')
    parser.add_argument("--generalize", action="store_true", help='eval generalization in contexts')
    parser.add_argument("--state_pred", action="store_true", help='use state prediction')
    parser.add_argument("--causal", action="store_true", help='use causal state prediction')

    return parser


if __name__ == '__main__':
    args = get_train_parser().parse_args()
    eval_avg_reward, eval_avg_succ, eval_avg_cost, eval_avg_crash, eval_avg_max_step, eval_avg_oor, eval_avg_over_speed = [], [], [], [], [], [], []
    model_est = CUDA(StatePred(hidden_dim=64, output_dim=35, causal=args.causal))
    if args.causal: 
        model_est.load_state_dict(torch.load('./state_est_causal.pt'))
    else: 
        model_est.load_state_dict(torch.load('./state_est.pt'))
    
    if args.method == 'expert': 
        pass
    elif args.method == 'ssr': 
        model = CUDA(SafeDecisionTransformer_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=args.hidden, context_len=args.context, n_heads=1, drop_p=0.1, max_timestep=args.horizon))
        model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'best.pt')))
        print(model)
    elif args.method == 'ssr_nc': 
        model = CUDA(DecisionTransformer(state_dim=35, act_dim=2, n_blocks=3, h_dim=args.hidden, context_len=args.context, n_heads=1, drop_p=0.1, max_timestep=args.horizon))
        model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'best.pt')))
    
    elif args.method == 'icil': 
        model = CUDA(ICIL(state_dim=35, action_dim=2, hidden_dim_input=64, hidden_dim=64))
        model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'hest.pt')))
    
    elif args.method == 'bnn': 
        model = CUDA(BNN_Agent(hidden_dim=64, action_dim=2))
        model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'hest.pt')))
    
    elif args.method == 'gsa': 
        model = CUDA(GSA_Agent(hidden_dim=64, action_dim=2, K=5))
        model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'hest.pt')))
    
    elif args.method == 'bc': 
        model = CUDA(BC_Agent(hidden_dim=64, action_dim=2))
        model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'hest.pt')))   
    
    for idx_trials in range(args.num_trials): 
        if args.method == 'expert': 
            env = SubprocVecEnv([lambda: make_envs_expert(idx_trials) for i in range(args.num_envs)])
        else:
            if args.generalize: 
                if args.single_env: 
                    env = SubprocVecEnv([lambda: make_envs_single(i, True) for i in range(args.num_envs)], start_method="spawn")    
                else: 
                    env = SubprocVecEnv([lambda: make_envs(True) for _ in range(args.num_envs)], start_method="spawn")
            else: 
                if args.single_env: 
                    env = SubprocVecEnv([lambda: make_envs_single(i) for i in range(args.num_envs)], start_method="spawn")    
                else: 
                    env = SubprocVecEnv([make_envs for _ in range(args.num_envs)], start_method="spawn")  
        
        if args.method == 'expert': 
            results = evaluate_expert(env, num_ep=args.ep_num)
            
        elif args.method == 'ssr': 
            if args.state_pred: 
                results = evaluate_on_env_structure(model, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=args.reward, ctg_target=args.cost, 
                                                            rtg_scale=300.0, ctg_scale=80.0, num_eval_ep=args.ep_num, max_test_ep_len=args.horizon, use_value_pred=args.value, 
                                                            use_state_pred=True, model_est=model_est)
        
            else: 
                results = evaluate_on_env_structure(model, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=args.reward, ctg_target=args.cost, 
                                                            rtg_scale=300.0, ctg_scale=80.0, num_eval_ep=args.ep_num, max_test_ep_len=args.horizon, use_value_pred=args.value)
        elif args.method == 'ssr_nc': 
            if args.state_pred: 
                results = evaluate_on_env_nocost(model, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=args.reward, ctg_target=args.cost, 
                                                            rtg_scale=300.0, ctg_scale=80.0, num_eval_ep=args.ep_num, max_test_ep_len=args.horizon, use_value_pred=args.value, 
                                                            use_state_pred=True, model_est=model_est)
        
            else: 
                results = evaluate_on_env_nocost(model, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=args.reward, ctg_target=args.cost, 
                                                            rtg_scale=300.0, ctg_scale=80.0, num_eval_ep=args.ep_num, max_test_ep_len=args.horizon, use_value_pred=args.value)
        
        # elif args.method == 'ssr_pred': 
        #     model = CUDA(SafeDecisionTransformer_Structure(state_dim=35, act_dim=2, n_blocks=3, h_dim=args.hidden, context_len=args.context, n_heads=1, drop_p=0.1, max_timestep=args.horizon))
        #     model.load_state_dict(torch.load(os.path.join('checkpoint/', args.model, 'best.pt')))
            
            
        #     results = evaluate_on_env_structure_pred(model, model_est, torch.device('cuda:0'), context_len=args.context, env=env, rtg_target=350, ctg_target=0., 
        #                                                 rtg_scale=40.0, ctg_scale=10.0, num_eval_ep=args.ep_num, max_test_ep_len=args.horizon)

        elif args.method == 'icil': 
            if args.state_pred: 
                results = eval_icil(model, torch.device('cuda:0'), env, num_eval_ep=args.ep_num, image=args.state_pred, model_est=model_est)    
            else: 
                results = eval_icil(model, torch.device('cuda:0'), env, num_eval_ep=args.ep_num, image=args.state_pred)    

        elif args.method == 'bnn': 
            results = eval_bnn(model, torch.device('cuda:0'), env, num_eval_ep=args.ep_num, image=False)

        elif args.method == 'gsa': 
            results = eval_gsa(model, torch.device('cuda:0'), env, num_eval_ep=args.ep_num, image=False)

        elif args.method == 'bc': 
            results = eval_bc(model, torch.device('cuda:0'), env, num_eval_ep=args.ep_num, image=False)
        
        elif args.method == 'cnn': 
            model = CUDA(BisimEncoder_Head_BP_Actor(hidden_dim=64, output_dim=2, causal=True))
            model.load_state_dict(torch.load('checkpoint/'+args.model+'.pt'))
            results = evaluate_on_env_cnn(model, torch.device('cuda:0'), env, num_eval_ep=50)
 
        elif args.method == 'bearl': 
            arg = BEARL_DEFAULT_CONFIG["MetaDrive-TopDown-v0"]()
            # model & optimizer setup
            model = BEARL(
                state_dim=env.observation_space['state'].shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                a_hidden_sizes=arg.a_hidden_sizes,
                c_hidden_sizes=arg.c_hidden_sizes,
                vae_hidden_sizes=arg.vae_hidden_sizes,
                sample_action_num=arg.sample_action_num,
                gamma=arg.gamma,
                tau=arg.tau,
                beta=arg.beta,
                lmbda=arg.lmbda,
                mmd_sigma=arg.mmd_sigma,
                target_mmd_thresh=arg.target_mmd_thresh,
                start_update_policy_step=arg.start_update_policy_step,
                num_q=arg.num_q,
                num_qc=arg.num_qc,
                PID=arg.PID,
                cost_limit=arg.cost_limit,
                episode_len=arg.episode_len,
                device=arg.device,
            )
            model.load_state_dict(torch.load(args.model)["model_state"])
            
            print("model loaded")
            results = evaluate_rollouts(model, env, num_eval_ep=args.ep_num)
        elif args.method == 'bcql': 
            arg = BCQL_DEFAULT_CONFIG["MetaDrive-TopDown-v0"]()
            # model & optimizer setup
            model = BCQL(
                state_dim=env.observation_space['state'].shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                a_hidden_sizes=arg.a_hidden_sizes,
                c_hidden_sizes=arg.c_hidden_sizes,
                vae_hidden_sizes=arg.vae_hidden_sizes,
                sample_action_num=arg.sample_action_num,
                PID=arg.PID,
                gamma=arg.gamma,
                tau=arg.tau,
                lmbda=arg.lmbda,
                beta=arg.beta,
                phi=arg.phi,
                num_q=arg.num_q,
                num_qc=arg.num_qc,
                cost_limit=arg.cost_limit,
                episode_len=arg.episode_len,
                device=arg.device,
            )
            model.load_state_dict(torch.load(args.model)["model_state"])
            
            print("model loaded")
            results = evaluate_rollouts(model, env, num_eval_ep=args.ep_num)
        
        elif args.method == 'cpq': 
            arg = CPQ_DEFAULT_CONFIG["MetaDrive-TopDown-v0"]()
            # model & optimizer setup
            model = CPQ(
                state_dim=env.observation_space["state"].shape[0],
                action_dim=env.action_space.shape[0],
                max_action=env.action_space.high[0],
                a_hidden_sizes=arg.a_hidden_sizes,
                c_hidden_sizes=arg.c_hidden_sizes,
                vae_hidden_sizes=arg.vae_hidden_sizes,
                sample_action_num=arg.sample_action_num,
                gamma=arg.gamma,
                tau=arg.tau,
                beta=arg.beta,
                num_q=arg.num_q,
                num_qc=arg.num_qc,
                qc_scalar=arg.qc_scalar,
                cost_limit=arg.cost_limit,
                episode_len=arg.episode_len,
                device=arg.device,
            )
            model.load_state_dict(torch.load(args.model)["model_state"])
            
            print("model loaded")
            results = evaluate_rollouts(model, env, num_eval_ep=args.ep_num)

        else: 
            raise NotImplementedError
        
        eval_avg_reward += [results['eval/avg_reward']]
        # eval_avg_ep_len += results['eval/avg_ep_len']
        eval_avg_succ += [results['eval/success_rate']]
        eval_avg_crash += [results['eval/crash_rate']]
        eval_avg_oor += [results['eval/oor_rate']]
        eval_avg_max_step += [results['eval/max_step']]
        eval_avg_over_speed += [results['eval/over_speed']]
        eval_avg_cost += [results['eval/avg_cost']]
        env.close()
    
    log_str = ("method: {}, checkpoint: {}".format(args.method, args.model) + '\n' + 
        "=" * 60 + '\n' +
        "eval avg reward: {:.2f}\scriptsize".format(np.mean(eval_avg_reward))+str("{")+"$\pm${:.2f}".format(np.std(eval_avg_reward))+str("}") + '\n' + 
        "eval avg succ: {:.2f}\scriptsize".format(np.mean(eval_avg_succ))+str("{")+"$\pm${:.2f}".format(np.std(eval_avg_succ))+str("}") + '\n' + 
        "eval avg crash: {:.2f}\scriptsize".format(np.mean(eval_avg_crash))+str("{")+"$\pm${:.2f}".format(np.std(eval_avg_crash))+str("}") + '\n' + 
        "eval avg oor: {:.2f}\scriptsize".format(np.mean(eval_avg_oor))+str("{")+"$\pm${:.2f}".format(np.std(eval_avg_oor))+str("}") + '\n' + 
        "eval avg overtime: {:.2f}\scriptsize".format(np.mean(eval_avg_max_step))+str("{")+"$\pm${:.2f}".format(np.std(eval_avg_max_step))+str("}") + '\n' + 
        "eval avg overspeed: {:.2f}\scriptsize".format(np.mean(eval_avg_over_speed))+str("{")+"$\pm${:.2f}".format(np.std(eval_avg_over_speed))+str("}") + '\n' + 
        "eval avg cost: {:.2f}\scriptsize".format(np.mean(eval_avg_cost))+str("{")+"$\pm${:.2f}".format(np.std(eval_avg_cost))+str("}") + '\n'
        )
    print(log_str)