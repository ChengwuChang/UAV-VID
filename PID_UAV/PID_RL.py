import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class PID_RL(BaseAviary):

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 target_position=None
                 ):

        # 初始化 base class 前先儲存必要屬性
        self._init_drone_model = drone_model
        self._init_num_drones = num_drones

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder)

        self.ctrl = [DSLPIDControl(drone_model=self.DRONE_MODEL) for _ in range(self.NUM_DRONES)]
        self.target_position = target_position if target_position is not None else np.array([[0.0, 0.0, 1.0] for _ in range(self.NUM_DRONES)])

    def set_target_position(self, target_position):
        self.target_position = target_position

    def _actionSpace(self):
        # action = 每台 drone 調整 PID 的 (P_x, P_y, P_z)
        act_lower_bound = np.array([[0.0, 0.0, 0.0] for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[5.0, 5.0, 5.0] for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        obs_lower_bound = np.array([[-np.inf]*20 for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf]*20 for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    def _preprocessAction(self, action):
        rpm = np.zeros((self.NUM_DRONES, 4))
        for i in range(self.NUM_DRONES):
            self.ctrl[i].P_COEFF_FOR = np.clip(action[i], 0.0, 5.0)
            state = self._getDroneStateVector(i)
            rpm[i, :], _, _ = self.ctrl[i].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=self.target_position[i]
            )
        return rpm

    def _computeReward(self):
        positions = np.array([self._getDroneStateVector(i)[:3] for i in range(self.NUM_DRONES)])
        error = np.linalg.norm(positions - self.target_position, axis=1)
        return -np.mean(error)

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {}
