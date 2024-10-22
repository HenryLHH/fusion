
import random
from metadrive.constants import HELP_MESSAGE

import matplotlib.pyplot as plt

from metadrive.component.vehicle_module.PID_controller import PIDController, Target

import numpy as np
import time as time
from tqdm import trange
import pickle


class Expert(object):
    def __init__(self, vehicle, speed=30) -> None:
        self.vehicle = vehicle
        self.target = Target(0.375, speed)
        self.steering_controller = PIDController(1.6, 0.0008, 27.3)
        self.acc_controller = PIDController(0.1, 0.001, 0.3)
        self.steering_error = None
        self.acc_error = None

    def get_action(self, o):
        if self.steering_error is None:
            self.steering_error = -self.target.lateral
            self.acc_error = self.vehicle.speed - self.target.speed
            self.steering = self.steering_controller.get_result(self.steering_error)
            self.acc = self.acc_controller.get_result(self.acc_error)
            return np.array([-self.steering, self.acc])
        else:
            self.steering_error = o[0] - self.target.lateral
            t_speed = self.target.speed if abs(o[12] - 0.5) < 0.01 else self.target.speed - 10
            self.acc_error = self.vehicle.speed - t_speed
            self.steering = self.steering_controller.get_result(self.steering_error)
            self.acc = self.acc_controller.get_result(self.acc_error)
            return np.array([-self.steering, self.acc])
        
        
      