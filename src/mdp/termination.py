import numpy as np


class TerminationChecker:
    def __init__(self, cfg):
        self.z_min = float(cfg["z_range"][0])
        self.z_max = float(cfg["z_range"][1])
        self.roll_min = np.deg2rad(cfg["roll_range"][0])
        self.roll_max = np.deg2rad(cfg["roll_range"][1])
        self.pitch_min = np.deg2rad(cfg["pitch_range"][0])
        self.pitch_max = np.deg2rad(cfg["pitch_range"][1])

    def is_healthy(self, state):
        return (
            np.isfinite(state).all()
            and self.z_min <= state[2] <= self.z_max
            and self.roll_min <= state[4] <= self.roll_max
            and self.pitch_min <= state[5] <= self.pitch_max
        )
