import argparse
import copy
import imp
from typing import Dict

import numpy as np
from metadrive import MetaDriveEnv, TopDownMetaDrive, SafeMetaDriveEnv
from envs.envs import State_TopDownMetaDriveEnv
from envs.vector_metadrive import RealState_TopDownMetaDriveEnv
from metadrive.manager.traffic_manager import TrafficMode


try:
    import ray
    from ray import tune

    from ray.tune import CLIReporter
    from ray.rllib.agents.callbacks import DefaultCallbacks
    from ray.rllib.env import BaseEnv
    from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
    from ray.rllib.policy import Policy
    # from ray.rllib.agents.ppo.ppo import PPOTrainerTorch
    # from ray.rllib.agents.ppo.ppo_torch_model import *
except ImportError:
    ray = None
    raise ValueError("Please install ray through 'pip install ray'.")

# from agent.bisim.ppo_lag_trainer import PPOLag
# from ray.rllib.agents.ppo import PPOTrainer

from ssr.agent.ppo_lag.ppo_lag_trainer import PPOLag



class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["cost"] = []

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["cost"].append(info["cost"])

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        **kwargs
    ):
        arrive_dest = episode.last_info_for()["arrive_dest"]
        crash = episode.last_info_for()["crash"]
        out_of_road = episode.last_info_for()["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["cost"] = np.nan
        if "custom_metrics" not in result:
            return

        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]

# @ray.remote(num_gpus=0.1)
# def use_gpu():
#     import torch
#     import os
#     print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
#     print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
#     print(torch.cuda.is_available())

def train(
    trainer,
    config,
    stop,
    exp_name,
    num_gpus=0,
    test_mode=False,
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    custom_callback=None,
    max_failures=1,
    **kwargs
):
    ray.init(num_gpus=num_gpus, _memory=int(2e10), _redis_max_memory=int(5e9), object_store_memory=int(5e9))
    print('ray get GPU: ', ray.get_gpu_ids())
    input()
    used_config = {
        "callbacks": custom_callback if custom_callback else DrivingCallbacks,  # Must Have!
    }
    used_config.update(config)
    config = copy.deepcopy(used_config)

    if not isinstance(stop, dict) and stop is not None:
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if keep_checkpoints_num is not None and not test_mode:
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
    progress_reporter = CLIReporter(metric_columns=metric_columns)
    progress_reporter.add_metric_column("success")
    progress_reporter.add_metric_column("crash")
    progress_reporter.add_metric_column("out")
    progress_reporter.add_metric_column("max_step")
    progress_reporter.add_metric_column("length")
    progress_reporter.add_metric_column("cost")
    kwargs["progress_reporter"] = progress_reporter

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True if "checkpoint_at_end" not in kwargs else kwargs.pop("checkpoint_at_end"),
        stop=stop,
        config=config,
        max_failures=max_failures if not test_mode else 0,
        reuse_actors=False,
        local_dir=".",
        **kwargs
    )
    return analysis


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="generalization_experiment")
    parser.add_argument("--num-gpus", type=int, default=0)
    return parser


if __name__ == '__main__':
    args = get_train_parser().parse_args()
    exp_name = args.exp_name
    stop = int(200_0000)
    config = dict(
        # ===== Training Environment =====
        # Train the policies in scenario sets with different number of scenarios.
        env=State_TopDownMetaDriveEnv, # MetaDriveEnv, # RealState_TopDownMetaDriveEnv, # State_TopDownMetaDriveEnv, #A State_TopDownMetaDriveEnv #MetaDriveEnv
        env_config=dict(
            environment_num=1000, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
            start_seed=0, #tune.grid_search([0, 1000]),
            frame_stack=3, # TODO: debug
            safe_rl_env=True,
            random_traffic=False,
            accident_prob=0,
            vehicle_config=dict(lidar=dict(
                num_lasers=240,
                distance=50,
                num_others=4
            )),
            traffic_density=0.2, #tune.grid_search([0.05, 0.2]),
            traffic_mode=TrafficMode.Hybrid,
            # IDM_agent=True,
            # resolution_size=64,
            # generalized_blocks=tune.grid_search([['X', 'T']])
        ),

        # ===== Evaluation =====
        # Evaluate the trained policies in unseen 200 scenarios.
        evaluation_interval=2,
        evaluation_num_episodes=40,
        metrics_smoothing_episodes=200,
        evaluation_config=dict(
            env_config=dict(
                environment_num=100, 
                start_seed=0, # .grid_search([0, 1000]), 
                frame_stack=3,
                safe_rl_env=True,
                random_traffic=False,
                accident_prob=0,
                vehicle_config=dict(lidar=dict(
                    num_lasers=240,
                    distance=50,
                    num_others=4
                )),
                traffic_density=0.2,
                traffic_mode=TrafficMode.Hybrid,
                # IDM_agent=True,
                # resolution_size=64,
                # generalized_blocks=tune.grid_search([['S', 'r', 'R', 'T'], ['C', 'X', 'O']]),
                )),
        evaluation_num_workers=10,
        
        # Hyper-parameters for PPO-Lag
        penalty_lr=0.01,
        cost_limit=20,
        # bisim_coef=0.0, # tune.grid_search([0.0, 0.1]),
        # model = dict(
        #     custom_model="model_img",
        # ),
        # ===== Training =====
        # Hyper-parameters for PPO
        horizon=1000,
        rollout_fragment_length=200,
        sgd_minibatch_size=100,
        train_batch_size=8000,
        num_sgd_iter=20,
        lr=5e-5, 
        # grad_clip=0.5,
        # entropy_coeff=0.005, 
        # kl_target=0.05,  # 0.05
        # clip_param=0.2, # 0.3
        num_workers=20,

        # model=tune.grid_search([dict(
        #         custom_model="my_torch_model",
        #         # Extra kwargs to be passed to your model's c'tor.
        #         # custom_model_config={"vf_share_layers": True},
        #         )]), #,  {"vf_share_layers": True}]),
        **{"lambda": 0.95},
        # ===== Resources Specification =====
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.5,
        num_cpus_for_driver=0.5,
        framework='torch',
    )
    
    train(
        PPOLag, #PPOTrainer, #PPOLag, # PPOTrainer,
        exp_name=exp_name,                                                                                                                                                                                                                                                                                                
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
    )



    # config = dict(
    #     env=State_TopDownMetaDriveEnv,
    #     env_config=dict(
    #         environment_num=1000, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
    #         start_seed=tune.grid_search([0, 1000]),
    #         frame_stack=1,
    #         random_traffic=False,
    #         safe_rl_env=True,
    #         accident_prob=0,
    #         traffic_density=0.2,
    #     ),
    #     # ===== Evaluation =====
    #     # Evaluate the trained policies in unseen 200 scenarios.
    #     evaluation_interval=2,
    #     evaluation_num_episodes=40,
    #     metrics_smoothing_episodes=200,
    #     evaluation_config=dict(env_config=dict(environment_num=1000, start_seed=2000)),
    #     evaluation_num_workers=1,

    #     # ===== Training =====
    #     optimization=dict(actor_learning_rate=1e-4, critic_learning_rate=1e-4, entropy_learning_rate=1e-4),
    #     prioritized_replay=False,
    #     horizon=1000,
    #     target_network_update_freq=1,
    #     timesteps_per_iteration=1000,
    #     learning_starts=10000,
    #     clip_actions=False,
    #     normalize_actions=True,
    #     num_cpus_for_driver=1,
    #     # No extra worker used for learning. But this config impact the evaluation workers.
    #     num_cpus_per_worker=0.5,
    #     # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
    #     num_workers=64,
    #     num_gpus=0.5 if args.num_gpus != 0 else 0
    # )
    # train(
    #     "SAC",
    #     exp_name=exp_name,
    #     keep_checkpoints_num=5,
    #     stop=stop,
    #     config=config,
    #     num_gpus=args.num_gpus,
    # )

