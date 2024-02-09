from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass
from pyrallis import field


@dataclass
class BEARLTrainConfig:
    # wandb params
    project: str = "FUSION-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BEARL"
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
    beta: float = 0.5
    lmbda: float = 0.75
    mmd_sigma: float = 50
    target_mmd_thresh: float = 0.05
    num_samples_mmd_match: int = 10
    start_update_policy_step: int = 0
    kernel: str = "gaussian" # or "laplacian"
    num_q: int = 2
    num_qc: int = 2
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)
    # evaluation params
    eval_episodes: int = 50
    eval_every: int = 500
    single_env: bool = False

@dataclass
class BEARLCarCircleConfig(BEARLTrainConfig):
    pass


@dataclass
class BEARLAntRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class BEARLDroneCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class BEARLCarRunConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class BEARLAntCircleConfig(BEARLTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500

@dataclass
class BEARLMetaDriveConfig(BEARLTrainConfig):
    # training params
    task: str = "MetaDrive-TopDown-v0"
    episode_len: int = 1000
    eval_episodes: int = 50
    single_env: bool = False

BEARL_DEFAULT_CONFIG = {
    "OfflineCarCircle-v0": BEARLCarCircleConfig,
    "OfflineAntRun-v0": BEARLAntRunConfig,
    "OfflineDroneRun-v0": BEARLDroneRunConfig,
    "OfflineDroneCircle-v0": BEARLDroneCircleConfig,
    "OfflineCarRun-v0": BEARLCarRunConfig,
    "OfflineAntCircle-v0": BEARLAntCircleConfig,
    "MetaDrive-TopDown-v0": BEARLMetaDriveConfig
}
