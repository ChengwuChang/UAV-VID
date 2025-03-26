import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class PID_RL(BaseAviary):


    ################################################################################

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
                 output_folder='results'
                 ):

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,#更新頻率要降低比較訓練好
                         ctrl_freq=ctrl_freq,#更新頻率要降低
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0., 0., 0., 0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array(
            [[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0., -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf,
                                     -np.inf, -np.inf, -np.inf, -np.inf, 0., 0., 0., 0.] for i in
                                    range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf, np.inf, np.inf, 1., 1., 1., 1., np.pi, np.pi, np.pi, np.inf, np.inf,
                                     np.inf, np.inf, np.inf, np.inf, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM,
                                     self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        """計算當前的獎勵值"""
        reward = 0.0

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)

            # 1. 位置誤差懲罰
            target_pos = np.array([0.0, 0.0, 1.0])  # 目標位置 (可變)
            pos = state[0:3]
            pos_error = np.linalg.norm(pos - target_pos)  # 位置誤差
            reward -= pos_error * 5.0  # 越接近目標獎勵越高

            # 2. 姿態誤差懲罰
            target_rpy = np.array([0.0, 0.0, 0.0])  # 理想姿態 (roll, pitch, yaw)
            rpy = state[7:10]
            rpy_error = np.linalg.norm(rpy - target_rpy)
            reward -= rpy_error * 2.0  # 避免過大角度偏移

            # 3. 速度懲罰
            vel = state[10:13]
            reward -= np.linalg.norm(vel) * 0.5  # 限制速度太快

            # 4. 能量消耗 (RPM 輸出) 懲罰
            motor_rpm = state[16:20]
            reward -= np.sum(motor_rpm) * 1e-5  # 減少不必要的動力輸出
        return -1

    ################################################################################

    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
    #     terminated = False
    #
    # for i in range(self.NUM_DRONES):
    #     state = self._getDroneStateVector(i)
    #
    #     # 高度檢查（低於 10cm 視為墜落）
    #     if state[2] < 0.1:
    #         terminated = True
    #
    #     # 姿態檢查（過大角度可能代表失控）
    #     if abs(state[7]) > np.radians(60) or abs(state[8]) > np.radians(60):
    #         terminated = True
    #
    #     # 速度過快（防止異常輸出）
    #     if np.linalg.norm(state[10:13]) > 10.0:
    #         terminated = True
    #
    #     return terminated
        return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        """返回當前訓練的資訊"""
        info = {}

        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            target_pos = np.array([0.0, 0.0, 1.0])
            pos_error = np.linalg.norm(state[0:3] - target_pos)
            rpy_error = np.linalg.norm(state[7:10])
            speed = np.linalg.norm(state[10:13])

            info[f'drone_{i}_pos_error'] = pos_error
            info[f'drone_{i}_rpy_error'] = rpy_error
            info[f'drone_{i}_speed'] = speed

        # return info
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years
