from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class BCQLTrainConfig:
    # wandb params
    project: str = "FUSION-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BNN"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    # training params
    task: str = "MetaDrive-TopDown-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cuda:0"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.0001
    critic_lr: float = 0.0001
    vae_lr: float = 0.0001
    phi: float = 0.05
    lmbda: float = 0.75
    beta: float = 0.5
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    num_q: int = 2
    num_qc: int = 2
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)
    # evaluation params
    eval_episodes: int = 50
    eval_every: int = 2500
    single_env: bool = False

@dataclass
class BCQLMetaDriveConfig(BCQLTrainConfig):
    # training params
    task: str = "MetaDrive-TopDown-v0"
    episode_len: int = 1000
    eval_episodes: int = 50
    


BCQL_DEFAULT_CONFIG = {
    "MetaDrive-TopDown-v0": BCQLMetaDriveConfig, 
}