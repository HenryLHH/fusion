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

from expert_controller import Expert

class State_TopDownObservation(TopDownMultiChannel):
    def __init__(self,
        vehicle_config,
        env,
        clip_rgb: bool,
        frame_stack: int = 5,
        post_stack: int = 5,
        frame_skip: int = 5,
        resolution=None,
        max_distance=50):
        
        # self.state_observor = StateObservation(vehicle_config)        
        # self.img_observor = TopDownMultiChannel(vehicle_config, env, clip_rgb, \
        #     frame_stack, post_stack, frame_skip, resolution, max_distance)
        TopDownMultiChannel.__init__(self, vehicle_config, env, clip_rgb, \
            frame_stack, post_stack, frame_skip, resolution, max_distance)
        

    @property
    def observation_space(self):
        shape = (self.num_stacks, ) + self.obs_shape
        # shape = (self.img_observor.num_stacks, ) + self.img_observor.obs_shape
        if self.rgb_clip:
            # return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
            return gym.spaces.Dict({
                "img": gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32),
                "state": gym.spaces.Box(-0.0, 1.0, shape=(35, ), dtype=np.float32),
                "expert": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)})
            
        else:
            # return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
            return gym.spaces.Dict({
                "img": gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8),
                "state": gym.spaces.Box(-0.0, 1.0, shape=(35, ), dtype=np.float32),
                "expert": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)})
    
    def observe(self, vehicle):
        # img = self.img_observor.observe(vehicle)
        # state = self.state_observor.observe(vehicle)
        img = TopDownMultiChannel.observe(self, vehicle)
        # state = StateObservation.observe(self, vehicle)
        return {'img': img.transpose((2, 0, 1))}

        
class State_TopDownMetaDriveEnv(TopDownMetaDrive): 
    # Safe features
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
        return State_TopDownObservation(
            self.config["vehicle_config"],
            self,
            self.config["rgb_clip"],
            frame_stack=self.config["frame_stack"],
            post_stack=self.config["post_stack"],
            frame_skip=self.config["frame_skip"],
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"]
        )
        
    def __init__(self, config):
        super(State_TopDownMetaDriveEnv, self).__init__(config)
        self.episode_cost = 0
        self.count_out_road = 0

    def _calc_dist(self, pos1, pos2, ori1, ori2, l1, l2, w1, w2):
        '''
            Calculate the distance w.r.t. base vehicles
        '''
        dist_wo_ori = norm(pos2[0]-pos1[0], pos2[1]-pos1[1])
        theta = np.arctan2(pos2[1]-pos1[1], pos2[0]-pos1[0])
        # print('Calc: ', l1 / 2./ np.sin(theta-ori1), w1 / 2./ np.cos(theta-ori1))
        dist_1 = min(abs(l1 / 2./ np.sin(theta-ori1)), abs(w1 / 2./ np.cos(theta-ori1)))
        dist_2 = min(abs(l2 / 2./ np.sin(np.pi-theta+ori2)), abs(w2 / 2./ np.cos(np.pi-theta+ori2)))
        
        return dist_wo_ori - dist_1 - dist_2      

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

    # def cost_function(self, vehicle_id: str):
    #     vehicle = self.vehicles[vehicle_id]
    #     step_info = dict()
    #     step_info["cost"] = 0
    #     if self._is_out_of_road(vehicle):
    #         step_info["cost"] = self.config["out_of_road_cost"]
    #     elif vehicle.crash_vehicle:
    #         step_info["cost"] = self.config["crash_vehicle_cost"]
    #     elif vehicle.crash_object:
    #         step_info["cost"] = self.config["crash_object_cost"]
    #     return step_info['cost'], step_info

    def done_function(self, vehicle_id: str):
        done, done_info = super(State_TopDownMetaDriveEnv, self).done_function(vehicle_id)    
        vehicle = self.vehicles[vehicle_id]
        
        try:
            lidar_state = self.lidar_state_obs.observe(vehicle)
            # true_state = self.state_obs.observe(vehicle)
        except:
            self.lidar_state_obs = LidarStateObservation(self.config["vehicle_config"])
            lidar_state = self.lidar_state_obs.observe(vehicle)

            # self.state_obs = StateObservation(self.config["vehicle_config"])
            # true_state = self.state_obs.observe(vehicle)            
        current_lane = vehicle.navigation.get_current_lane(vehicle)[1]
        done_info.update({"true_state": lidar_state[:35], 'current_lane': current_lane})
        # done_info.update({"true_state": true_state, 'current_lane': current_lane})
        
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

    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        if self.expert is None:
            self.expert = Expert(self.vehicles[DEFAULT_AGENT], speed=100)
            self.expert_action = self.expert.get_action(0)

        o, r, d, i = super(State_TopDownMetaDriveEnv, self).step(actions)
        o.update({'state': i['true_state']})
        self.expert_action = self.expert.get_action(o['state'])

        o.update({'expert': np.clip(self.expert_action, -np.ones(2,), np.ones(2))})
        
        
        # add noise level
        epsilon = np.random.normal(np.zeros_like(o['img']), \
            self.config['obs_noise_scale']*np.ones_like(o['img']))
        o['img'] += epsilon
        o['img'] = np.clip(o['img'], -np.zeros_like(o['img']), np.ones_like(o['img']))
        
        return o, r, d, i

    
    def reset(self, *args, **kwargs):
        self.episode_cost = 0
        self.count_out_road = 0
        self.expert = None
        self.expert_action = np.zeros(2,)
        # self.expert_action = self.expert.get_action(0)
        o = super(State_TopDownMetaDriveEnv, self).reset(*args, **kwargs)
        # self.expert = Expert(self.vehicles[DEFAULT_AGENT])
        o.update({'state': self.done_function(DEFAULT_AGENT)[1]['true_state'], 'expert': self.expert_action})
        return o

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
        return StateObservation(self.config["vehicle_config"])
    
    @property
    def observation_space(self) -> gym.Space:
        shape = 19
        return gym.spaces.Box(-0.0, 1.0, shape=(shape, ), dtype=np.float32)


    def reset(self, *args, **kwargs):
        self.episode_cost = 0
        self.count_out_road = 0
        self.expert = None
        super(RealState_TopDownMetaDriveEnv, self).reset(*args, **kwargs)
        return self.done_function(DEFAULT_AGENT)[1]['true_state']
     
    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        o, r, d, i = super(RealState_TopDownMetaDriveEnv, self).step(actions)
        o = i['true_state']
        epsilon = np.random.normal(np.zeros_like(o), self.config['obs_noise_scale']*np.ones_like(o))
        o = o + epsilon
        o = np.clip(o, np.zeros_like(o), np.ones_like(o))
        return o, r, d, i

    def cost_function(self, vehicle_id: str):

        vehicle = self.vehicles[vehicle_id]
        
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] += self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] += self.config["crash_object_cost"]
        
        self.episode_cost += step_info["cost"]
        step_info["total_cost"] = self.episode_cost
        return step_info["cost"], step_info
        
    def done_function(self, vehicle_id: str):
        done, done_info = super(RealState_TopDownMetaDriveEnv, self).done_function(vehicle_id)    
        vehicle = self.vehicles[vehicle_id]
        
        try:
            lidar_state = self.lidar_state_obs.observe(vehicle)
            # true_state = self.state_obs.observe(vehicle)
        except:
            self.lidar_state_obs = LidarStateObservation(self.config["vehicle_config"])
            lidar_state = self.lidar_state_obs.observe(vehicle)
            # self.state_obs = StateObservation(self.config["vehicle_config"])
            # true_state = self.state_obs.observe(vehicle)            
        current_lane = vehicle.navigation.get_current_lane(vehicle)[1]
        done_info.update({"true_state": lidar_state[:19], 'current_lane': current_lane})
        # done_info.update({"true_state": true_state, 'current_lane': current_lane})
        
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
    env = State_TopDownMetaDriveEnv({'traffic_density': 0.3, 'vehicle_config': {'lidar': {'num_lasers': 240, 'distance': 50, 'num_others': 4}}})
    # print(env.observation_space)
    print(env.config)
    o = env.reset()
    # print(o)
    expert_action = np.zeros(2,)
    print(env.observation_space['state'].shape)
    # print(env.observation_space)
    
    for _ in range(1000):
        o, r, d, i = env.step(np.clip(expert_action, -np.ones(2,), np.ones(2)))
        # env.render(mode="top_down", film_size=(800, 800))
        expert_action = o['expert']
        print(np.clip(i['raw_action'], -np.ones(2,), np.ones(2,)))
        print(r)
        print(o['state'][19:])
        print('=====')
        print(np.clip(expert_action, -np.ones(2,), np.ones(2,)))
        
        