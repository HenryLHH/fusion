from typing import Union, Dict, AnyStr, Optional, Tuple

from metadrive import TopDownMetaDrive
from metadrive.obs.state_obs import StateObservation, LidarStateObservation
from metadrive.utils import Config
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.utils.math_utils import norm, clip
# from metadrive.obs.top_down_obs import TopDownObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.obs.observation_base import ObservationBase
from metadrive.component.vehicle_module.navigation import Navigation


import numpy as np
import gym


class StateObservation_VehicleInfo(StateObservation):

    @property
    def observation_space(self):
        return gym.spaces.Box(-0.0, 1.0, shape=(35, ), dtype=np.float32)
    
    def observe(self, vehicle):
        obs_state = super().observe(vehicle)    
        objs = vehicle.lidar.get_surrounding_objects(vehicle)
        
        obs_others = vehicle.lidar.get_surrounding_vehicles_info(vehicle, objs)

        return np.concatenate([obs_state, obs_others], axis=0)


class RealState_TopDownMetaDriveEnv(TopDownMetaDrive): 
    @classmethod
    def default_config(cls) -> Config:
        config = TopDownMetaDrive.default_config()
        config.update({
            "accident_prob": 0.0,
            "safe_rl_env": True,
            "crash_vehicle_cost": 1,
            "crash_object_cost": 1,
            "out_of_road_cost": 1.,  # only give penalty for out_of_road
            "use_lateral": False,
            "distance": 20, # same with offline data
        })
        return config

    def get_single_observation(self, _=None):
        return StateObservation_VehicleInfo(self.config["vehicle_config"])

    def reset(self, *args, **kwargs):
        self.episode_cost = 0
        self.count_out_road = 0

        o = super(RealState_TopDownMetaDriveEnv, self).reset(*args, **kwargs)
        return o
        
    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        
        return super(RealState_TopDownMetaDriveEnv, self).step(actions)

    def cost_function(self, vehicle_id: str):

        vehicle = self.vehicles[vehicle_id]
        
        step_info = dict()
        step_info["cost"] = 0
        # if self._is_out_of_road(vehicle):
        #     step_info["cost"] = self.config["out_of_road_cost"]
        if vehicle.crash_vehicle:
            step_info["cost"] += self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] += self.config["crash_object_cost"]
        
        self.episode_cost += step_info["cost"]
        step_info["total_cost"] = self.episode_cost
        return step_info["cost"], step_info
        
    def done_function(self, vehicle_id: str):
        done, done_info = super(RealState_TopDownMetaDriveEnv, self).done_function(vehicle_id)    
        vehicle = self.vehicles[vehicle_id]
        current_lane = vehicle.navigation.get_current_lane(vehicle)[1]
        done_info.update({'current_lane': current_lane})
        
        if self.config["safe_rl_env"]:
            if done_info[TerminationState.CRASH_VEHICLE]:
                done = False
            elif done_info[TerminationState.CRASH_OBJECT]:
                done = False
            if vehicle_id == DEFAULT_AGENT:
                if done_info[TerminationState.OUT_OF_ROAD]:
                    if self.count_out_road <= 10:
                        self.count_out_road += 1
                        # done_info[TerminationState.OUT_OF_ROAD]=False
                        done = False
                    else:
                        done = True

        return done, done_info
    
if __name__ == '__main__':
    env = RealState_TopDownMetaDriveEnv({'traffic_density': 0.3, 'vehicle_config': {'lidar': {'num_lasers': 1, 'distance': 50}}})
    print(env.observation_space)
    o = env.reset()
    print('reset: ', o)
    input()
    expert_action = np.zeros(2,)
    print(env.observation_space)
    input()
    for _ in range(1000):
        o, r, d, i = env.step(np.random.rand(2,))
        # env.render(mode="top_down", film_size=(800, 800))
        print(i)
        # print(r)
        # print('=====')
        # print(o)
        
        