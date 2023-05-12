import numpy as np
import gym
from gym.spaces import Discrete
from typing import Dict, List, Optional

# from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.misc import normc_initializer

from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel

from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType

torch, nn = try_import_torch()

SCALE_DIAG_MIN_MAX = (-20, 2)


class ConstrainedSACModel(SACTorchModel):
    """Extension of standard TFModel for SAC.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        policy_model_config: ModelConfigDict = None,
        q_model_config: ModelConfigDict = None,
        twin_q: bool = False,
        initial_alpha: float = 1.0,
        target_entropy: Optional[float] = None,
    ):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation (str): activation for actor network
            actor_hiddens (list): hidden layers sizes for actor network
            critic_hidden_activation (str): activation for critic network
            critic_hiddens (list): hidden layers sizes for critic network
            twin_q (bool): build twin Q networks.
            initial_alpha (float): The initial value for the to-be-optimized
                alpha parameter (default: 1.0).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of SACModel.
        """
        obs_space = obs_space.original_space['img']

        super(ConstrainedSACModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.discrete = False
        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
            action_outs = q_outs = self.action_dim
        else:
            self.action_dim = np.product(action_space.shape)
            action_outs = 2 * self.action_dim
            q_outs = 1

        self.cost_q_net = self.build_q_model(self.obs_space, self.action_space)
        self.cost_twin_q_net = self.build_q_model(self.obs_space, self.action_space)
        self.register_variables(self.q_net.variables)
        self.register_variables(self.cost_q_net.variables)

        if twin_q:
            self.twin_q_net = build_q_net(
                "twin_q", self.model_out, self.actions_input
            )
            self.register_variables(self.twin_q_net.variables)
        else:
            self.twin_q_net = None
        if twin_cost_q:
            self.cost_twin_q_net = build_q_net(
                "cost_twin_q", self.model_out, self.actions_input
            )
            self.register_variables(self.cost_twin_q_net.variables)
        else:
            self.cost_twin_q_net = None

        self.log_alpha = tf.Variable(
            np.log(initial_alpha), dtype=tf.float32, name="log_alpha"
        )
        self.alpha = tf.exp(self.log_alpha)

        # Auto-calculate the target entropy.
        if target_entropy is None or target_entropy == "auto":
            # See hyperparams in [2] (README.md).
            if self.discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / action_space.n), dtype=np.float32
                )
            # See [1] (README.md).
            else:
                target_entropy = -np.prod(action_space.shape)
        self.target_entropy = target_entropy

        self.register_variables([self.log_alpha])


    def get_cost_q_values(self, model_out, actions=None):
        if actions is not None:
            return self.cost_q_net([model_out, actions])
        else:
            return self.cost_q_net(model_out)

    def get_twin_cost_q_values(self, model_out, actions=None):
        if actions is not None:
            return self.cost_twin_q_net([model_out, actions])
        else:
            return self.cost_twin_q_net(model_out)


    def get_policy_output(self, model_out):
        """Return the action output for the most recent forward pass.

        This outputs the support for pi(s). For continuous action spaces, this
        is the action directly. For discrete, is is the mean / std dev.

        Arguments:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].

        Returns:
            tensor of shape [BATCH_SIZE, action_out_size]
        """
        return self.action_model(model_out)

    def policy_variables(self):
        """Return the list of variables for the policy net."""

        return list(self.action_model.variables)

    def q_variables(self):
        """Return the list of variables for Q / twin Q nets."""

        return self.q_net.variables + (
            self.twin_q_net.variables if self.twin_q_net else []
        )

    def cost_q_variables(self):
        return self.cost_q_net.variables + (
            self.cost_twin_q_net.variables
            if self.cost_twin_q_net else []
        )
