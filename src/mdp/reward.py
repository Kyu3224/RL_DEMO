import numpy as np


class RewardCalculator:
    def __init__(self, cfg):
        self.reward_weights = {
            k: float(v)
            for k, v in cfg["reward"].items()
        }
        self.cost_weights = {
            k: float(v)
            for k, v in cfg["cost"].items()
        }

        self._tracking_velocity_sigma = cfg["command"]["tracking_velocity_sigma"]

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)

    def reset(self):
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)

    # Reward Methods Defined
    def linear_velocity(self, desired, current):
        vel_sqr_error = np.sum(
            np.square(desired - current)
        )
        return self.reward_weights["linear_vel_tracking"] * np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    def angular_velocity(self, desired, current):
        vel_sqr_error = np.square(desired - current)
        return self.reward_weights["angular_vel_tracking"] * np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    def healthy(self, is_healthy):
        return self.reward_weights["healthy"] * float(is_healthy)

    def feet_air_time(self, feet_contact_forces, dt, desired):
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        feet_contact_force_mag = feet_contact_forces
        curr_contact = feet_contact_force_mag > 1.0
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 1.0) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(desired) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return self.reward_weights["feet_air_time"] * air_time_reward

    # Cost Methods Defined
    def torque_cost(self, torques):
        return self.cost_weights["torque"] * np.sum(np.square(torques))

    def action_rate(self, last_actions, actions):
        return self.cost_weights["action_rate"] * np.sum(np.square(last_actions - actions))

    def vertical_vel(self, vertical_vel):
        return self.cost_weights["vertical_vel"] * np.square(vertical_vel)

    def xy_angular_velocity(self, ang_vel):
        return self.cost_weights["xy_angular_vel"] * np.sum(np.square(ang_vel))

    def joint_limit(self, soft_joint_range, jpos):
        # Penalize the robot for joints exceeding the soft control range
        out_of_range = (soft_joint_range[:, 0] - jpos).clip(
            min=0.0
        ) + (jpos - soft_joint_range[:, 1]).clip(min=0.0)
        return self.cost_weights["joint_limit"] * np.sum(out_of_range)

    def joint_acc_limit(self, qacc):
        return self.cost_weights["joint_acc"] * np.sum(np.square(qacc))

