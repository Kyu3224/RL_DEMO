class Go1State:
    def __init__(self, data):
        self.data = data

    @property
    def base_lin_vel(self):
        return self.data.qvel[:3]

    @property
    def base_ang_vel(self):
        return self.data.qvel[3:6]

    @property
    def joint_pos(self):
        return self.data.qpos[7:]

    @property
    def joint_vel(self):
        return self.data.qvel[6:]

    @property
    def joint_acc(self):
        return self.data.qacc[6:]

    @property
    def joint_trq(self):
        return self.data.qfrc_actuator[-12:]

