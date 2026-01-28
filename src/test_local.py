import time
import yaml
from pathlib import Path

import mujoco

if not hasattr(mujoco.MjData, 'solver_iter'):
    setattr(mujoco.MjData, 'solver_iter', property(lambda self: self.solver_niter))

from stable_baselines3 import PPO
from go1_mujoco_env import Go1MujocoEnv

MODEL_DIR = "models"
LOG_DIR = "logs"

def test():
    # model_path = "/home/kdyun/Desktop/RL_DEMO/models/pretrained/best_model.zip"
    # model_path = "/home/kdyun/Desktop/RL_DEMO/models/2026-01-28_20-59-34/best_model.zip"
    # model_path = "/home/kdyun/Desktop/RL_DEMO/models/2026-01-28_21-29-16/best_model.zip"
    # model_path = "/home/kdyun/Desktop/RL_DEMO/models/2026-01-28_21-52-50/best_model.zip"
    # model_path = "/home/kdyun/Desktop/RL_DEMO/models/2026-01-28_23-09-06/best_model.zip"
    # model_path = "/home/kdyun/Desktop/RL_DEMO/models/2026-01-28_23-18-10/best_model.zip"
    model_path = "/home/kdyun/Desktop/RL_DEMO/models/2026-01-28_23-37-48/best_model.zip"
    # model_path = "/home/kdyun/Desktop/RL_DEMO/models/2026-01-28_23-58-40/best_model.zip"
    model_path = Path(model_path)

    cfg_path = Path(f"/home/kdyun/Desktop/RL_DEMO/src/params.yaml")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Set a fixed command for testing [vx (m/s), vy (m/s), wz (rad/s)]
    given_command = [0.7, 0.0, 0.0]

    env = Go1MujocoEnv(
        prj_path="/home/kdyun/Desktop/RL_DEMO",
        given_command=given_command,  # Use fixed command instead of random
        render_mode="human",
        camera_name="tracking",
        width=1920,
        height=1080,
    )

    model = PPO.load(path=model_path, env=env, verbose=1)

    max_time_step_s = cfg["test"]["max_time_step_s"]
    total_reward = 0
    total_length = 0
    if total_length < int(max_time_step_s * 500):
        obs, _ = env.reset()
        env.render()

        ep_len = 0
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1

            # Slow down the rendering
            time.sleep(0.02)

            if terminated or truncated:
                print(f"{ep_len=}  {ep_reward=}")
                break

        total_length += ep_len
        total_reward += ep_reward



if __name__ == "__main__":
    test()
