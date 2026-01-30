from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

import mujoco

import numpy as np
from pathlib import Path

import yaml

from src.utils import math_utils
from src.utils.state import Go1State
from src.mdp.reward import RewardCalculator
from src.mdp.termination import TerminationChecker

DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}


class Go1MujocoEnv(MujocoEnv):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, prj_path, given_command=None, **kwargs):
        model_path = Path(f"{prj_path}/unitree_go1/scene_position.xml")
        cfg_path = Path(f"{prj_path}/src/envs.yaml")

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Store fixed command if provided (for testing)
        self._given_command = given_command

        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # Perform an action every 10 frames (dt(=0.002) * 10 = 0.02 seconds -> 50hz action rate)
            observation_space=None,  # Manually set afterwards
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Update metadata to include the render FPS
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": cfg["render"]["render_fps"],
        }
        self._last_render_time = -1.0
        self._max_episode_time_sec = cfg["env"]["max_episode_length_s"]
        self._step = 0

        # Weights for the reward and cost functions
        self.reward_calculator = RewardCalculator(cfg=cfg)

        self._curriculum_base = cfg["curriculum"]["base"]
        self._gravity_vector = np.array(self.model.opt.gravity)
        self._default_joint_position = np.array(self.model.key_ctrl[0])
        # print("self._default_joint_position   " ,self._default_joint_position )
        # vx (m/s), vy (m/s), wz (rad/s)
        self._desired_velocity_min = np.array(cfg["command"]["des_vel"]["min"])
        self._desired_velocity_max = np.array(cfg["command"]["des_vel"]["max"])
        self._desired_velocity = self._sample_desired_vel()
        self._obs_scale = {
            k: float(v)
            for k, v in cfg["observation"].items()
        }

        # Metrics used to determine if the episode should be terminated
        self.termination = TerminationChecker(cfg=cfg["termination"])

        self._cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL
        self._cfrc_ext_contact_indices = [2, 3, 5, 6, 8, 9, 11, 12]

        # Non-penalized degrees of freedom range of the control joints
        dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        ctrl_range_offset = (
                0.5
                * (1 - dof_position_limit_multiplier)
                * (
                        self.model.actuator_ctrlrange[:, 1]
                        - self.model.actuator_ctrlrange[:, 0]
                )
        )
        # First value is the root joint, so we ignore it
        self._soft_joint_range = np.copy(self.model.actuator_ctrlrange)
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0.1

        # Action: 12 torque values
        self._last_action = np.zeros(12)

        # Gait phase tracking
        self._phase = 0.0
        self._gait_hz = 1  # Gait frequency (Hz)
        self._phase_sin = np.zeros(2)  # [sin(phase), cos(phase)]

        # Foot contact phase for gait enforcement
        self._foot_contact_phase = np.zeros(4)  # [RR, RL, FR, FL]

        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float32
        )

        # Feet site names to index mapping
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-site
        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtobj
        feet_site = [
            "FR",
            "FL",
            "RR",
            "RL",
        ]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }

        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )

        # Re-initialize action space now that _default_joint_position is defined
        self._set_action_space()

    def _set_action_space(self):
        """Override parent's action space to use relative joint offsets instead of absolute positions."""
        # Action space is now relative offsets from nominal pose
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T

        # Get default joint position (nominal pose)
        default_pos = self.model.key_ctrl[0].copy().astype(np.float32)

        # Action space: relative offsets from nominal pose
        self.action_space = spaces.Box(
            low=low - default_pos,
            high=high - default_pos,
            dtype=np.float32
        )
        return self.action_space

    def step(self, action):
        self._step += 1
        # Convert relative action (from nominal pose) to absolute joint position
        absolute_action = action + self._default_joint_position
        # print("absolute_action   " ,absolute_action )

        self.do_simulation(absolute_action, self.frame_skip)

        # Update gait phase
        self._update_gait_phase()

        observation = self._get_obs()
        reward, reward_info = self._get_reward(action)
        terminated = not self.is_healthy
        truncated = self._step >= (self._max_episode_time_sec / self.dt)

        # Add termination penalty if robot falls
        if terminated:
            reward -= 100.0  # Large penalty for falling
            reward_info["cost/termination"] = 100.0
        else:
            reward_info["cost/termination"] = 0.0

        infos = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
                1.0 / self.metadata["render_fps"]
        ):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action

        return observation, reward, terminated, truncated, infos

    @property
    def is_healthy(self):
        return self.termination.is_healthy(self.state_vector())

    @property
    def projected_gravity(self):
        w, x, y, z = self.data.qpos[3:7]
        euler_orientation = np.array(math_utils.euler_from_quaternion(w, x, y, z))
        projected_gravity_not_normalized = (
                np.dot(self._gravity_vector, euler_orientation) * euler_orientation
        )
        if np.linalg.norm(projected_gravity_not_normalized) == 0:
            return projected_gravity_not_normalized
        else:
            return projected_gravity_not_normalized / np.linalg.norm(
                projected_gravity_not_normalized
            )

    @property
    def feet_contact_forces(self):
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)

    @property
    def feet_positions_z(self):
        """Get Z positions of feet relative to ground (height above ground)."""
        feet_z = np.zeros(4)
        feet_site_names = ["FR", "FL", "RR", "RL"]
        for i, site_name in enumerate(feet_site_names):
            site_id = self._feet_site_name_to_id[site_name]
            feet_z[i] = self.data.site_xpos[site_id][2]
        return feet_z

    @property
    def curriculum_factor(self):
        return self._curriculum_base ** 0.997

    def _get_reward(self, action):
        state = Go1State(data=self.data)
        # Positive Rewards
        linear_vel_tracking_reward = self.reward_calculator.linear_velocity(desired=self._desired_velocity[:2], current=state.base_lin_vel[:2])
        angular_vel_tracking_reward = self.reward_calculator.angular_velocity(desired=self._desired_velocity[2], current=state.base_ang_vel[2])
        healthy_reward = self.reward_calculator.healthy(is_healthy=self.is_healthy)
        base_height_reward = self.reward_calculator.base_height(robot_height=state.base_pos[2], ref_height=0.3)
        feet_air_time_reward = self.reward_calculator.feet_air_time(feet_contact_forces=self.feet_contact_forces, dt=self.dt, desired=self._desired_velocity[:2])

        rewards = sum([
            linear_vel_tracking_reward,
            angular_vel_tracking_reward,
            healthy_reward,
            base_height_reward,
            feet_air_time_reward,
        ])

        # Negative Costs
        ctrl_cost = self.reward_calculator.torque_cost(torques=state.joint_trq)
        action_rate_cost = self.reward_calculator.action_rate(last_actions=self._last_action, actions=action)
        vertical_vel_cost = self.reward_calculator.vertical_vel(vertical_vel=state.base_lin_vel[2])
        xy_angular_vel_cost = self.reward_calculator.xy_angular_velocity(ang_vel=state.base_ang_vel[:2])
        joint_limit_cost = self.reward_calculator.joint_limit(soft_joint_range=self._soft_joint_range, jpos=state.joint_pos)
        joint_acc_cost = self.reward_calculator.joint_acc_limit(qacc=state.joint_acc)
        action_norm_cost = self.reward_calculator.action_norm(action=action)
        joint_pos_deviation_cost = self.reward_calculator.joint_pos_deviation(jpos=state.joint_pos, nominal_jpos=self._default_joint_position)

        # Gait-related costs
        gait_enforcement_cost = self.reward_calculator.gait_enforcement(
            foot_contact_forces=self.feet_contact_forces,
            foot_contact_phase=self._foot_contact_phase
        )
        foot_clearance_cost = self.reward_calculator.foot_clearance(
            foot_positions=self.feet_positions_z,
            foot_contact_phase=self._foot_contact_phase
        )

        costs = sum([
            ctrl_cost,
            action_rate_cost,
            vertical_vel_cost,
            xy_angular_vel_cost,
            joint_limit_cost,
            joint_acc_cost,
            action_norm_cost,
            joint_pos_deviation_cost,
            gait_enforcement_cost,
            foot_clearance_cost,
        ])

        reward = rewards - costs

        reward_info = {
            "reward/total": reward,
            "reward/total_rewards": rewards,
            "reward/total_costs": costs,
            "reward/lin_vel": linear_vel_tracking_reward,
            "reward/ang_vel": angular_vel_tracking_reward,
            "reward/feet_air_time": feet_air_time_reward,
            "reward/base_height_reward": base_height_reward,
            "reward/healthy": healthy_reward,
            "cost/torque": ctrl_cost,
            "cost/action_rate": action_rate_cost,
            "cost/vertical_vel": vertical_vel_cost,
            "cost/xy_angular_vel": xy_angular_vel_cost,
            "cost/joint_lim": joint_limit_cost,
            "cost/joint_acc": joint_acc_cost,
            "cost/action_norm": action_norm_cost,
            "cost/joint_pos_deviation": joint_pos_deviation_cost,
            "cost/gait_enforcement": gait_enforcement_cost,
            "cost/foot_clearance": foot_clearance_cost,
        }

        return reward, reward_info

    def _get_obs(self):
        # The first three indices are the global x,y,z position of the trunk of the robot
        # The second four are the quaternion representing the orientation of the robot
        # The above seven values are ignored since they are privileged information
        # The remaining 12 values are the joint positions
        # The joint positions are relative to the starting position
        dofs_position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]

        # The first three values are the global linear velocity of the robot
        # The second three are the angular velocity of the robot
        # The remaining 12 values are the joint velocities
        velocity = self.data.qvel.flatten()
        base_linear_velocity = velocity[:3]
        base_angular_velocity = velocity[3:6]
        dofs_velocity = velocity[6:]

        desired_vel = self._desired_velocity
        last_action = self._last_action
        projected_gravity = self.projected_gravity

        curr_obs = np.concatenate(
            (
                base_linear_velocity * self._obs_scale["linear_velocity"],
                base_angular_velocity * self._obs_scale["angular_velocity"],
                projected_gravity,
                desired_vel * self._obs_scale["linear_velocity"],
                dofs_position * self._obs_scale["dofs_position"],
                dofs_velocity * self._obs_scale["dofs_velocity"],
                last_action,
                self._phase_sin,  # Add gait phase (sin, cos) encoding (2D)
            )
        ).clip(-self._clip_obs_threshold, self._clip_obs_threshold)

        return curr_obs

    def reset_model(self):
        # Reset the position and control values with noise
        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
        self.data.ctrl[:] = self.model.key_ctrl[
                                0
                            ] + self._reset_noise_scale * self.np_random.standard_normal(
            *self.data.ctrl.shape
        )

        # Reset the variables and sample a new desired velocity
        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(12)
        self._last_render_time = -1.0

        # Reset gait phase (randomize start phase like RaiSim)
        if self.np_random.uniform() <= 0.5:
            self._phase = 0.0
        else:
            self._phase = self._gait_hz / 2.0
        self._foot_contact_phase = np.zeros(4)

        # Reset buffer at reward calculator
        self.reward_calculator.reset()

        observation = self._get_obs()

        return observation

    def _update_gait_phase(self):
        """Update gait phase and compute foot contact phases for trot gait."""
        # Update phase
        self._phase += self.dt

        # Compute phase sin/cos for observation
        phase_val = self._phase / self._gait_hz * 2 * np.pi
        self._phase_sin[0] = np.sin(phase_val)
        self._phase_sin[1] = np.cos(phase_val)

        # Compute foot contact phase for each foot (trot gait pattern)
        # Order: [FR, FL, RR, RL] based on _cfrc_ext_feet_indices = [4, 7, 10, 13]
        # RaiSim order was: [RR, RL, FR, FL]
        # We need to map: FR(0), FL(1), RR(2), RL(3)

        base_phase = np.sin(phase_val)
        self._foot_contact_phase[2] = base_phase      # RR
        self._foot_contact_phase[3] = -base_phase     # RL (opposite)
        self._foot_contact_phase[0] = -base_phase     # FR (opposite)
        self._foot_contact_phase[1] = base_phase      # FL (same as RR)

    def _sample_desired_vel(self):
        # If given_command is provided, use it instead of random sampling
        if self._given_command is not None:
            return np.array(self._given_command)

        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return desired_vel
