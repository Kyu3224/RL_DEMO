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
        self._feet_air_time += dt * (~curr_contact)

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 0.5) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(desired) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return self.reward_weights["feet_air_time"] * air_time_reward

    ### Add your own reward ###
    def base_height(self, robot_height, ref_height):
        """
        Design a reward that keeps the robot's base height close to z_ref.

        Hints:
        - Try L2 penalty:        -(robot_height - ref_height)^2
        - Try exponential form: exp(-(robot_height - ref_height)^2 / sigma)
        - Think about tolerance vs strictness
        """
        base_height_reward = 0.0  # TODO: implement your reward here
        return self.reward_weights["base_height"] * base_height_reward

    # Cost(Negative Rewards) Methods Defined
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

    def action_norm(self, action):
        """Penalize action magnitude (deviation from nominal pose)"""
        return self.cost_weights["action_norm"] * np.sum(np.square(action))

    def joint_pos_deviation(self, jpos, nominal_jpos):
        """Penalize actual joint position deviation from nominal pose"""
        return self.cost_weights["joint_pos_deviation"] * np.sum(np.square(jpos - nominal_jpos))

    def termination(self, is_terminated):
        """Large penalty for falling/termination"""
        return self.cost_weights["termination"] * float(is_terminated)

    def gait_enforcement(self, foot_contact_forces, foot_contact_phase):
        """
        Enforce trot gait pattern using quadratic penalty.
        Penalizes when actual foot contact doesn't match the desired phase.

        Args:
            foot_contact_forces: Contact forces for each foot [FR, FL, RR, RL]
            foot_contact_phase: Desired phase for each foot [-1, 1]

        Returns:
            Cost (penalty) for gait mismatch
        """
        # Compute actual contact state (binary)
        curr_contact = foot_contact_forces > 1.0  # [FR, FL, RR, RL]

        # Compute foot_contact_double as in RaiSim
        # If in contact: positive phase value, if in air: negative phase value
        foot_contact_double = np.zeros(4)
        for i in range(4):
            if curr_contact[i]:
                foot_contact_double[i] = 1.0 * foot_contact_phase[i]
            else:
                foot_contact_double[i] = -1.0 * foot_contact_phase[i]

        # ===== Option 1: Quadratic penalty (currently used) =====
        penalty = np.sum((foot_contact_double - 1.0) ** 2)
        return self.cost_weights["gait_enforcement"] * penalty

        # ===== Option 2: Relaxed log barrier (original RaiSim) =====
        # # Apply relaxed log barrier to keep values in range [-0.6, 2.0]
        # limit_lower = -0.6
        # limit_upper = 2.0
        # delta = 0.1
        #
        # barrier_reward = 0.0
        # for i in range(4):
        #     barrier_reward += self._relaxed_log_barrier(
        #         delta, limit_lower, limit_upper, foot_contact_double[i]
        #     )
        #
        # # Clip to prevent extreme gradients (as in RaiSim line 442-448)
        # barrier_reward = max(barrier_reward, -300.0)
        #
        # # Return as cost (negative reward)
        # return self.cost_weights["gait_enforcement"] * (-barrier_reward)

    def foot_clearance(self, foot_positions, foot_contact_phase):
        """
        Enforce foot clearance during swing phase using quadratic penalty.
        Ensures feet lift high enough during swing phase.

        Args:
            foot_positions: Z-positions of feet relative to ground [FR, FL, RR, RL] (4,)
            foot_contact_phase: Desired phase for each foot [-1, 1]

        Returns:
            Cost (penalty) for insufficient clearance
        """
        desired_foot_clearance = 0.1 

        # ===== Option 1: Quadratic penalty (currently used) =====
        # Penalize deviation from desired clearance during swing phase
        penalty = 0.0
        for i in range(4):
            # Only enforce clearance during swing phase (phase < -0.6)
            if foot_contact_phase[i] < -0.6:
                # Penalize when foot is below desired clearance
                clearance_error = foot_positions[i] - desired_foot_clearance
                if clearance_error < 0:  # Below desired height
                    penalty += clearance_error ** 2

        return self.cost_weights["foot_clearance"] * penalty

        # ===== Option 2: Relaxed log barrier (original RaiSim) =====
        # limit_lower = -0.08
        # limit_upper = 1.0
        # delta = 0.02
        #
        # foot_clearance_vals = np.zeros(4)
        # for i in range(4):
        #     # Only enforce clearance during swing phase (phase < -0.6)
        #     if foot_contact_phase[i] < -0.6:
        #         foot_clearance_vals[i] = foot_positions[i] - desired_foot_clearance
        #     else:
        #         foot_clearance_vals[i] = 0.0  # Max reward (not enforcing)
        #
        # barrier_reward = 0.0
        # for i in range(4):
        #     barrier_reward += self._relaxed_log_barrier(
        #         delta, limit_lower, limit_upper, foot_clearance_vals[i]
        #     )
        #
        # # Clip to prevent extreme gradients (as in RaiSim)
        # barrier_reward = max(barrier_reward, -300.0)
        #
        # # Return as cost (negative reward)
        # return self.cost_weights["foot_clearance"] * (-barrier_reward)

    @staticmethod
    def _relaxed_log_barrier(delta, alpha_lower, alpha_upper, x):
        """
        Relaxed log barrier function as in RaiSim.
        Returns positive reward when x is within bounds, negative outside.

        Args:
            delta: Relaxation parameter
            alpha_lower: Lower bound
            alpha_upper: Upper bound
            x: Value to check

        Returns:
            Barrier reward (positive inside bounds)
        """
        reward = 0.0

        # Lower bound
        x_temp = x - alpha_lower
        if x_temp < delta:
            reward += 0.5 * ((x_temp - 2*delta) / delta)**2 - 1 - np.log(delta)
        else:
            reward += -np.log(x_temp)

        # Upper bound
        x_temp = -(x - alpha_upper)
        if x_temp < delta:
            reward += 0.5 * ((x_temp - 2*delta) / delta)**2 - 1 - np.log(delta)
        else:
            reward += -np.log(x_temp)

        # Return positive reward (multiply by -1 to flip sign)
        return -reward

