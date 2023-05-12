from typing import Dict, Union, Optional, Type, List

import numpy as np

import ray
from ray.rllib.algorithms.ppo.ppo_tf_policy import validate_config as original_validate_config
from ray.rllib.algorithms.ppo.ppo import PPO

from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.torch_mixins import ValueNetworkMixin, KLCoeffMixin, EntropyCoeffSchedule, LearningRateSchedule
# from ray.rllib.agents.ppo.ppo_tf_policy import postprocess_ppo_gae
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation import postprocessing as rllib_post
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import _get_shared_metrics
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, StandardizeFields, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.torch_utils import apply_grad_clipping, explained_variance, sequence_mask

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import AgentID, TensorType, TrainerConfigDict
from ray.util.iter import LocalIterator
from ray.rllib.utils.annotations import override


# from .ppo_lag_model_torch import CostValueNetwork, CostValueNetworkMixin
# from .model_state import CostValueNetwork, CostValueNetworkMixin
# from .model_raw import CostValueNetwork, CostValueNetworkMixin
# from .model_bisim import CostValueNetwork, CostValueNetworkMixin
# from .model_gru import CostValueNetwork, CostValueNetworkMixin
# from .model_icil import CostValueNetwork, CostValueNetworkMixin
from .model_joint import CostValueNetwork, CostValueNetworkMixin

if hasattr(rllib_post, "discount_cumsum"):
    discount = rllib_post.discount_cumsum
else:
    discount = rllib_post.discount
Postprocessing = rllib_post.Postprocessing
torch, nn = try_import_torch()

COST_ADVANTAGE = "cost_advantage"
COST = "cost"
COST_LIMIT = "cost_limit"
COST_TARGET = "cost_target"
COST_VALUES = "cost_values"
PENALTY_LR = "penalty_lr"



def compute_cost_advantages(rollout: SampleBatch, last_r: float, gamma: float = 0.9, lambda_: float = 1.0):
    # print('compute_cost_advantages:', rollout[COST_VALUES].shape, np.array([last_r]).shape)
    vpred_t = np.concatenate([rollout[COST_VALUES], np.array([last_r])])
    delta_t = (rollout[COST] + gamma * vpred_t[1:] - vpred_t[:-1])
    rollout[COST_ADVANTAGE] = discount(delta_t, gamma * lambda_)
    rollout[COST_TARGET] = (rollout[COST_ADVANTAGE] + rollout[COST_VALUES]).copy().astype(np.float32)
    rollout[COST_ADVANTAGE] = rollout[COST_ADVANTAGE].copy().astype(np.float32)
    return rollout


def postprocess_ppo_cost(policy: Policy, sample_batch: SampleBatch) -> SampleBatch:
    # Trajectory is actually complete -> last r=0.0.
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch["state_out_{}".format(i)][-1])
        last_r = policy._cost_value(
            sample_batch[SampleBatch.NEXT_OBS][-1], sample_batch[SampleBatch.ACTIONS][-1], sample_batch[COST][-1],
            *next_state
        )
        last_r = last_r.detach().cpu().numpy()
    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.

    batch = compute_cost_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"])

    return batch

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



# def validate_config(config):
#     original_validate_config(config)
#     assert config["batch_mode"] == "complete_episodes", "We need to compute episode cost!"


class UpdatePenaltyMixin:

    def __init__(self):
        self._penalty_optimizer = None
        self._penalty_loss = 0.

    def update_penalty(self, batch):
        if self._penalty_optimizer is None:
            self._penalty_optimizer = torch.optim.Adam([self._penalty_param], lr=self.config[PENALTY_LR])        

        # self._penalty_loss = 0.

        ep_cost = (batch[COST].sum() / max(1., batch[SampleBatch.DONES].sum()))
        penalty_loss = -self._penalty_param * (ep_cost.mean() - self.config[COST_LIMIT])
        
        self._penalty_optimizer.zero_grad()
        penalty_loss.backward()
        self._penalty_optimizer.step()

        self._penalty_loss = penalty_loss.item()
        return self._penalty_loss


class PPOLagTorchPolicy(
    CostValueNetworkMixin,
    PPOTorchPolicy,
    UpdatePenaltyMixin
):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):
        CostValueNetworkMixin.__init__(self, observation_space, action_space, config)
        ########### PPO_LAG ################
        penalty_init = torch.tensor(0.0)
        penalty_param = nn.parameter.Parameter(penalty_init, requires_grad=True)
        self._penalty_active_fn = nn.Softplus()
        self._penalty = self._penalty_active_fn(penalty_param)
        self._penalty_param = penalty_param
        
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        UpdatePenaltyMixin.__init__(self)
        # print('init penalty: ', self._penalty_param)
    
    @override(PPOTorchPolicy)
    def make_model(self):
        dist_class, logit_dim = ModelCatalog.get_action_dist(self.action_space, self.config["model"])
        return ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=logit_dim,
            model_config=self.config["model"],
            framework="torch",
            model_interface=CostValueNetwork
        )

    def vf_preds_fetches(self, input_dict, state_batches, model, action_dist) -> Dict[str, TensorType]:
        return {
            SampleBatch.VF_PREDS: self.model.value_function(),
            COST_VALUES: self.model.get_cost_value(),
        }
    
    @override(PPOTorchPolicy)
    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        with torch.no_grad():
            stats_dict = self.vf_preds_fetches(input_dict, state_batches, model, action_dist)
            return convert_to_numpy(stats_dict)

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for Proximal Policy Objective.

        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.

        Returns:
            The PPO loss tensor given the input batch.
        """

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid
        
        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )
        
        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # Only calculate kl loss if necessary (kl-coeff > 0.0).
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )
        mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        ############################
        cost_adv = train_batch[COST_ADVANTAGE]
        surrogate_cost = cost_adv * torch.clamp(logp_ratio, 0., 1 + self.config["clip_param"])
        mean_cost_loss = reduce_mean_valid(surrogate_cost)
        cost_value_loss = torch.square(model.get_cost_value() - train_batch[COST_TARGET])
        penalty = self._penalty_active_fn(self._penalty_param)
        ############################
        
        # Compute a value function loss.
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        # Ignore the value function.
        else:
            value_fn_out = 0
            vf_loss_clipped = mean_vf_loss = 0.0

        total_loss = reduce_mean_valid(
            -surrogate_loss + self.kl_coeff * action_kl
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        ) + penalty * mean_cost_loss

        ############################
        total_loss /= (1 + penalty)
        self._mean_cost_loss = mean_cost_loss
        self._mean_cost_value_loss = reduce_mean_valid(cost_value_loss)
        total_loss += self._mean_cost_value_loss
        ############################
        
        # Add mean_kl_loss (already processed through `reduce_mean_valid`),
        # if necessary.
        # if self.config["kl_coeff"] > 0.0:
        #     total_loss += self.kl_coeff * mean_kl_loss
        
        # debug: DAGGER
        # total_loss = model.loss_bc
        
        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = mean_policy_loss
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        ##################################
        model.tower_stats["penalty"] = penalty
        model.tower_stats["penalty_param"] = self._penalty_param
        model.tower_stats["cost_loss"] = self._mean_cost_loss
        model.tower_stats["cost_value_loss"] = self._mean_cost_value_loss
        ##################################

        return total_loss

    @override(PPOTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_kl_coeff": self.kl_coeff,
                "cur_lr": self.cur_lr,
                "total_loss": torch.mean(
                    torch.stack(self.get_tower_stats("total_loss"))
                ),
                "policy_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_policy_loss"))
                ),
                "vf_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_vf_loss"))
                ),
                "vf_explained_var": torch.mean(
                    torch.stack(self.get_tower_stats("vf_explained_var"))
                ),
                "kl": torch.mean(torch.stack(self.get_tower_stats("mean_kl_loss"))),
                "entropy": torch.mean(
                    torch.stack(self.get_tower_stats("mean_entropy"))
                ),
                "entropy_coeff": self.entropy_coeff,
                #########################
                "penalty": torch.mean(torch.stack(self.get_tower_stats("penalty"))), 
                "penalty_param": torch.mean(torch.stack(self.get_tower_stats("penalty_param"))), 
                "cost_loss": torch.mean(torch.stack(self.get_tower_stats("cost_loss"))), 
                "cost_value_loss": torch.mean(torch.stack(self.get_tower_stats("cost_value_loss"))), 
                #########################
            }
        )

    @override(PPOTorchPolicy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        # TODO: no_grad still necessary?
        with torch.no_grad():
            infos = sample_batch.get(SampleBatch.INFOS)
            if isinstance(sample_batch['infos'][0], dict):
                sample_batch[COST] = np.array([info["cost"] for info in infos])
            else:
                sample_batch[COST] = np.zeros_like(sample_batch[SampleBatch.REWARDS])

            sample_batch = postprocess_ppo_cost(
                self, sample_batch
            )
            return compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)