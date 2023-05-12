from metadrive import MetaDriveEnv, SafeMetaDriveEnv
from metadrive.envs.argoverse_env import ArgoverseEnv
from metadrive.constants import HELP_MESSAGE
from metadrive.component.map.base_map import BaseMap, MapGenerateMethod, parse_map_config
from metadrive.component.blocks.first_block import FirstPGBlock

safe_config = dict(
    # controller="joystick",
    use_render=False,
    manual_control=False,
    traffic_density=0.05,
    environment_num=100,
    random_agent_model=False,
    random_lane_width=False,
    random_lane_num=False,
    map=3,  # seven block
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: 3, 
        BaseMap.LANE_WIDTH: 3.0,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
    },
    safe_rl_env=True,
    accident_prob=0.,
    start_seed=0, #random.randint(0, 1000)
    vehicle_config = {"vehicle_model": "default",
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1), 
                "lidar": {"num_lasers": 240, "distance": 50, "num_others": 0, "gaussian_noise": 0.02, "dropout_prob": 0.1},
                "side_detector": {"num_lasers": 0, "distance": 50, "gaussian_noise": 0.02, "dropout_prob": 0.2},
                "lane_line_detector": {"num_lasers": 0, "distance": 20, "gaussian_noise": 0.05, "dropout_prob": 0.2}}
)


safe_config_eval = dict(
    use_render=False,
    manual_control=False,
    traffic_density=0.05,
    environment_num=50,
    random_agent_model=False,
    random_lane_width=True,
    random_lane_num=True,
    map=3,  # seven block
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: 3, 
        BaseMap.LANE_WIDTH: 3.0,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
    },
    safe_rl_env=True,
    accident_prob=0.,
    start_seed=0, #random.randint(0, 1000)
    vehicle_config = {"vehicle_model": "default",
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1)}
                # "lidar": {"num_lasers": 240, "distance": 50, "num_others": 0, "gaussian_noise": 0.02, "dropout_prob": 0.1},
                # "side_detector": {"num_lasers": 0, "distance": 50, "gaussian_noise": 0.02, "dropout_prob": 0.2},
                # "lane_line_detector": {"num_lasers": 0, "distance": 20, "gaussian_noise": 0.05, "dropout_prob": 0.2}}
)
