import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
#import sys

class pid_RL(BaseAviary):

    def __init__(self,
                 delta_bound=0.5,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones:   int         = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics:     Physics     = Physics.PYB,
                 pyb_freq:    int         = 240,
                 ctrl_freq:   int         = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 w_radius: float = 0.04,

                 # 這裡改成有預設值 None
                 target_traj: np.ndarray = None,
                 max_steps:    int        = 500,
                 pid_bounds:   dict       = None,
                 
                 
                 ):
        self.step_count = 0
        self.error_history = []
        # 先用 drone_model、num_drones 建一組「空殼」的 PID controllers
        #  由於你已經在 __init__ 收到 drone_model, num_drones
        self.ctrl = [
            DSLPIDControl(drone_model)
            for _ in range(num_drones)
        ]
        # 1) 強制檢查：一定要傳入 target_traj
        if target_traj is None:
            raise ValueError("初始化 pid_RL 時，必須用 keyword 傳入 target_traj (shape=(T,3))")

        # 2) 先把 PID range 的屬性設好，避免 super().__init__ 呼叫 _actionSpace 時找不到
        default_lo = np.hstack([
            DSLPIDControl(drone_model).P_COEFF_FOR,
            DSLPIDControl(drone_model).I_COEFF_FOR,
            DSLPIDControl(drone_model).D_COEFF_FOR
        ])
        default_hi = default_lo * 10
        self.pid_lo = pid_bounds["lo"] if pid_bounds else default_lo
        self.pid_hi = pid_bounds["hi"] if pid_bounds else default_hi
        self.w_radius = w_radius

        # 3) 先把理想軌跡與步數上限存起來
        self.target_traj = np.array(target_traj)
        self.max_steps   = max_steps
        self.INIT_Z = float(self.target_traj[0, 2])

        # 4) 呼叫父類，把 DRONE_MODEL、NUM_DRONES 等都設好
        super().__init__(
            drone_model=drone_model,
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
            output_folder=output_folder
        )

        # 1) 把原本 DSLPIDControl 的初始 PID 當作 baseline
        ctrl0 = DSLPIDControl(self.DRONE_MODEL)
        self.base_pid = np.hstack([
            ctrl0.P_COEFF_FOR,
            ctrl0.I_COEFF_FOR,
            ctrl0.D_COEFF_FOR
        ])           # shape = (9,)

        # 2) 動作空間：允許在 [-delta_bound, +delta_bound] 之間微調
        self.delta_bound = delta_bound
        self.action_space = spaces.Box(
            low = -delta_bound * np.ones_like(self.base_pid),
            high=  delta_bound * np.ones_like(self.base_pid),
            dtype=np.float32
        )
    def _actionSpace(self):
    # 只有 9 維，不要包 NUM_DRONES 維度
        return spaces.Box(
            low  = self.pid_lo,    # shape = (9,)
            high = self.pid_hi,    # shape = (9,)
         dtype= np.float32
        )

    def _observationSpace(self):
        # 觀察可以包含當前位置誤差(3)、速度誤差(3)、當前 PID 參數(6)
        shape = (self.NUM_DRONES, 3 + 3 + len(self.pid_lo))
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """
        接受 SB3 可能傳入的 seed/options，並呼叫父類 reset。
        最後回傳 (obs, info) 的 Gymnasium 格式。
        """
        # 1) 如果 BaseAviary.reset 支援 seed，則傳下去；否則就丟掉
        try:
            obs_raw, info_raw = super().reset(seed=seed, options=options)
        except TypeError:
            obs_raw, info_raw = super().reset()

        # 2) 清內部變數
        self.error_history.clear()
        self.step_count = 0
        for c in self.ctrl:
            c.reset()

        # 3) 回傳你自訂的 observation + empty info dict
        return self._computeObs(), {}

    def step(self, action):
    # 1) 把 action 轉成陣列、統一成 (batch, dim)
        a = np.array(action)
        if a.ndim == 1:
            a = a.reshape(1, -1)

        # 2) 直接把 a 交給 BaseAviary.step()：
        #    它會自動呼 _preprocessAction(a) 來算馬達 rpm
        obs_raw, reward_raw, terminated, truncated, info_raw = super().step(a)
        self.step_count += 1

        # 3) 用你自己定義的方法再包一次 obs/reward/info
        obs    = self._computeObs()
        reward = self._computeReward()
        info   = self._computeInfo() if (terminated or truncated) else {}

        return obs, reward, terminated, truncated, info


    def _preprocessAction(self, action):

        """
        action.shape = (N, 9) 代表每架 drone 各 9 維 Δpid
        回傳 rpm (N,4)
        """
        rpms = np.zeros((self.NUM_DRONES, 4))
        for i in range(self.NUM_DRONES):
            delta = np.clip(action[i], -self.delta_bound, self.delta_bound)
            pid_vals = self.base_pid + delta    # 真正的 P,I,D

            P = pid_vals[0:3]
            I = pid_vals[3:6]
            D = pid_vals[6:9]

            # 注入到 PID 控制器
            self.ctrl[i].P_COEFF_FOR = P
            self.ctrl[i].I_COEFF_FOR = I
            self.ctrl[i].D_COEFF_FOR = D

            # 以新的 PID 值算 rpm
            state  = self._getDroneStateVector(i)
            target = self.target_traj[min(self.step_count, len(self.target_traj)-1)]
            rpms[i], pos_e, _ = self.ctrl[i].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=target
            )
            self.error_history.append(np.linalg.norm(pos_e))
        return rpms

    def _computeObs(self):
        obs = []
        state_list = [self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]
        for i, state in enumerate(state_list):
            cur_pos = state[0:3]
            cur_vel = state[10:13]
            target  = self.target_traj[min(self.step_count, len(self.target_traj)-1)]
            pos_err = cur_pos - target
            vel_err = cur_vel  # 如果有目標速度，可做 cur_vel - target_vel
            pid_vals = np.hstack([
                self.ctrl[i].P_COEFF_FOR,
                self.ctrl[i].I_COEFF_FOR,
                self.ctrl[i].D_COEFF_FOR,
            ])
            obs.append(np.concatenate([pos_err, vel_err, pid_vals]))
        return np.array(obs, dtype=np.float32)

    def _computeReward(self):
        # 1) 先拿出當前狀態
        state  = self._getDroneStateVector(0)
        pos    = state[0:3]
        target = self.target_traj[min(self.step_count, len(self.target_traj)-1)]

        # 2) 基本誤差
        dist   = np.linalg.norm(pos - target)            # 3D 距離
        z_err  = abs(pos[2] - target[2])                 # 垂直高度誤差
        dz     = pos[2] - getattr(self, "prev_z", self.INIT_Z)  # Δz bonus
        self.prev_z = pos[2]

        # 3) 計算半徑誤差
        r_cur   = np.linalg.norm(pos[:2])                # 當前水平半徑
        r_ideal = np.linalg.norm(target[:2])             # 理想水平半徑 (就是 R)
        rad_err = abs(r_cur - r_ideal)

        reward = (
            - (dist**2.0)
            - 0.1 * z_err
            + 0.2 * max(dz, 0.0)
            - self.w_radius * rad_err
        )
        print(f"[DBG reward] dist²={dist**2:.3f}), "f"z_err={- 0.1 *z_err:.3f}")
        print(f"dz={+ 0.2 * max(dz, 0.0):.3f}, rad_err= {- self.w_radius * rad_err:.3f}")
        print(f"total_reward = {reward:.3f}")
        return float(reward)


    def _computeTerminated(self):
        # 例如：跌落或超出範圍就結束
        for i in range(self.NUM_DRONES):
            if self._getDroneStateVector(i)[2] < 0.1:
                return True
        return False

    def _computeTruncated(self):
        # 超過最長步數就截斷
        return self.step_count >= self.max_steps

    def _computeInfo(self):
        # episode 結束時計算 mae, rmse, std_error，但我感覺有跟沒有好像根本沒差，目前沒看到學長或是任何人寫過這個函式
        errors = np.array(self.error_history)
        return {
            "mae":       float(np.mean(errors)),
            "rmse":      float(np.sqrt(np.mean(errors**2))),
            "std_error": float(np.std(errors)),
        }
