from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
import os
import uuid

import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa
from stable_baselines3.common.vec_env import SubprocVecEnv


from utils.logger import WandbLogger
from utils.dataset import TransitionDataset_Baselines # TODO
from fusion.agent.cpq.cpq import CPQ, CPQTrainer
from utils.exp_utils import make_envs, make_envs_single, auto_name, seed_all

from fusion.configs.cpq_configs import CPQTrainConfig, CPQ_DEFAULT_CONFIG


@pyrallis.wrap()
def train(args: CPQTrainConfig):
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # setup logger
    args.episode_len = 1000
    cfg = asdict(args)
    default_cfg = asdict(CPQ_DEFAULT_CONFIG[args.task]())
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    # logger = TensorboardLogger(args.logdir, log_txt=True, name=args.name)
    logger.save_config(cfg, verbose=args.verbose)

    # wrapper
    # initialize environment
    if args.single_env: 
        env = SubprocVecEnv([lambda: make_envs_single(i) for i in range(16)], start_method="spawn")    
    else: 
        env = SubprocVecEnv([make_envs for _ in range(16)])

    # model & optimizer setup
    model = CPQ(
        state_dim=env.observation_space["state"].shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        a_hidden_sizes=args.a_hidden_sizes,
        c_hidden_sizes=args.c_hidden_sizes,
        vae_hidden_sizes=args.vae_hidden_sizes,
        sample_action_num=args.sample_action_num,
        gamma=args.gamma,
        tau=args.tau,
        beta=args.beta,
        num_q=args.num_q,
        num_qc=args.num_qc,
        qc_scalar=args.qc_scalar,
        cost_limit=args.cost_limit,
        episode_len=args.episode_len,
        device=args.device,
    )
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    def checkpoint_fn():
        return {"model_state": model.state_dict()}

    logger.setup_checkpoint_fn(checkpoint_fn)

    trainer = CPQTrainer(model,
                         env,
                         logger=logger,
                         actor_lr=args.actor_lr,
                         critic_lr=args.critic_lr,
                         alpha_lr=args.alpha_lr,
                         vae_lr=args.vae_lr,
                         reward_scale=args.reward_scale,
                         cost_scale=args.cost_scale,
                         device=args.device)
    
    # initialize environment
    # pre-process offline dataset
    data_path = os.path.join("dataset", args.dataset)
    dataset = TransitionDataset_Baselines(data_path)

    trainloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    trainloader_iter = iter(trainloader)

    # for saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    for step in trange(args.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        _, _, _, _, observations, next_observations, actions, rewards, costs, _, _, done = [
            b.to(args.device) for b in batch
        ]
        trainer.train_one_step(observations, next_observations, actions, rewards, costs,
                               done)
        # evaluation
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            results = trainer.evaluate(args.eval_episodes)
            cost = results["cost"]
            ret = results["reward"]
            length = results["ep_len"]
            success_rate = results["success"]
            oor_rate = results["oor"]
            crash_rate = results["crash"]
            max_time = results["max_time"]
            logger.store(tab="eval", Cost=cost, Reward=ret, Length=length, \
                        Success=success_rate, OOR=oor_rate, Crash=crash_rate, Overtime=max_time)
            # save the current weight
            logger.save_checkpoint()
            # save the best weight
            if cost < best_cost or (cost == best_cost and ret > best_reward):
                best_cost = cost
                best_reward = ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)

        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()
