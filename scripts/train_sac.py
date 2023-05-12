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
from ray.rllib.agents.sac import SACTrainer


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
        env=TopDownMetaDrive,
        env_config=dict(
            environment_num=1000, # tune.grid_search([1, 5, 10, 20, 50, 100, 300, 1000]),
            start_seed=tune.grid_search([0, 1000]),
            frame_stack=3,
            random_traffic=False,
            # safe_rl_env=True,
            accident_prob=0,
            traffic_density=0.2,
        ),
        # ===== Evaluation =====
        # Evaluate the trained policies in unseen 200 scenarios.
        evaluation_interval=2,
        evaluation_num_episodes=40,
        metrics_smoothing_episodes=200,
        evaluation_config=dict(env_config=dict(environment_num=1000, start_seed=2000)),
        evaluation_num_workers=1,

        # ===== Training =====
        optimization=dict(actor_learning_rate=5e-5, critic_learning_rate=5e-5, entropy_learning_rate=5e-5),
        # prioritized_replay=False,
        replay_buffer_config=dict(capacity=int(2e5)), 
        horizon=1000,
        target_network_update_freq=1,
        timesteps_per_iteration=1000,
        learning_starts=10000,
        clip_actions=True,
        grad_clip=0.5,
        normalize_actions=True,
        num_cpus_for_driver=1,
        # No extra worker used for learning. But this config impact the evaluation workers.
        num_cpus_per_worker=0.5,
        # num_gpus_per_worker=0.1 if args.num_gpus != 0 else 0,
        num_workers=64,
        num_gpus=0.5 if args.num_gpus != 0 else 0
    )
    
    train(
        SACTrainer, 
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
    )

