
import logging
from typing import List, Optional, Type, Union
import math

from ray.rllib.agents.ppo import PPOTrainer as PPO
from ray.util.debug import log_once
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    Deprecated,
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.execution.common import _get_shared_metrics
from ray.tune.utils.util import merge_dicts

logger = logging.getLogger(__name__)

COST_ADVANTAGE = "cost_advantage"
COST = "cost"
COST_LIMIT = "cost_limit"
COST_TARGET = "cost_target"
COST_VALUES = "cost_values"
PENALTY_LR = "penalty_lr"

PPO_LAG_CONFIG = merge_dicts(PPO.get_default_config(), {
    COST_LIMIT: 10.,  # Or 25, 50.
    PENALTY_LR: 1e-2,
    "batch_mode": "complete_episodes",
    "lr": 5e-5,
    # "num_sgd_iter": 10,
    # "train_batch_size": 30000,
    # "num_workers": 5,
})

class UpdatePenalty:
    def __init__(self, workers):
        self.workers = workers

    def __call__(self, batch):
        def update(pi, pi_id):
            res = pi.update_penalty(batch)
            return (pi_id, res)

        res = self.workers.local_worker().foreach_trainable_policy(update)

        metrics = _get_shared_metrics()
        metrics.info["penalty_loss"] = res[0][1]

        return batch  # , fetch


class PPOLag(PPO):
    @override(PPO)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:

        if config["framework"] == "torch":
            from .ppo_lag_torch import PPOLagTorchPolicy
            return PPOLagTorchPolicy
        
        elif config["framework"] == "tf":
            raise NotImplementedError

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return PPO_LAG_CONFIG

    @override(PPO)
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        if self._by_agent_steps:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config["train_batch_size"]
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
            )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages", COST_ADVANTAGE])
        # Update penalty, TODO
        for policy_id in train_batch.policy_batches:
            batch = train_batch.policy_batches[policy_id]
            ret = self.get_policy(policy_id).update_penalty(batch)
        #     metrics = _get_shared_metrics()
        #     metrics.info["penalty_loss"] = ret
        
        # train_batch = train_batch.for_each(UpdatePenalty(self.workers))


        # Train
        if self.config["simple_optimizer"]:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        # For each policy: update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config["vf_loss_coeff"] * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config["vf_clip_param"]
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results
