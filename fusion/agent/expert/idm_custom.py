from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.vehicle_module.PID_controller import PIDController
import numpy as np


class IDMPolicy_CustomSpeed(IDMPolicy): 
    def __init__(self, control_object, random_seed):
        super(IDMPolicy_CustomSpeed, self).__init__(control_object, random_seed)
        self.NORMAL_SPEED = self.engine.global_config.get("idm_target_speed", 30)
        # self.DISTANCE_WANTED = 10. - self.engine.global_config.get("idm_target_speed", 30) / 6.
        self.target_speed = self.engine.global_config.get("idm_target_speed", 30)
        self.ACC_FACTOR = self.engine.global_config.get("idm_acc_factor", 1.0)
        self.DEACC_FACTOR = self.engine.global_config.get("idm_deacc_factor", -5.0)
        
        self.LANE_CHANGE_FREQ = 1000 # disable lane changing
        self.enable_lane_change = False
        self.target_speed = self.NORMAL_SPEED
        self.heading_pid = PIDController(self.KP_HEADING, 0.01, 3.5)
        self.lateral_pid = PIDController(self.KP_LATERAL, .002, 0.05)
        
    
    def act(self, *args, **kwargs):
        [steering, acc] = super(IDMPolicy_CustomSpeed, self).act(*args, **kwargs)
        steering += 0.05*(np.random.rand()-0.5)
        acc += 0.05*(np.random.rand()-0.5)
        steering = np.clip(steering, -1., 1.)
        acc = np.clip(acc, -1., 1.)

        return [steering, acc]