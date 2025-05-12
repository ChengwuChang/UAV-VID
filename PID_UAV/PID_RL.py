import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
import pybullet as p
import os
import time
from PIL import Image
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
                 target_position=None):
        # 儲存建構前的參數（BaseAviary 中會用到）
        self._init_drone_model = drone_model
        self._init_num_drones = num_drones
        # if initial_xyzs is None:
        #     H = 0.1
        #     initial_xyzs = np.array([[0, 0, H] for _ in range(num_drones)])
        # if initial_rpys is None:
        #     initial_rpys = np.zeros((num_drones, 3))

        # 呼叫父類別初始化環境（建立場景、無人機等）
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

        # 初始化每台無人機的 DSLPID 控制器
        self.ctrl = [DSLPIDControl(drone_model=self.DRONE_MODEL) for _ in range(self.NUM_DRONES)]
        self.stuck_counter = np.zeros(self.NUM_DRONES, dtype=int)
        # 初始化在 __init__ 中
        self.wp_stay_counter = np.zeros(self.NUM_DRONES, dtype=int)

        # 預設目標位置為每台無人機在 (0, 0, 1)
        self.target_position = target_position if target_position is not None else np.array(
            [[0.0, 0.0, 1.0] for _ in range(self.NUM_DRONES)])
        self._init_path()


    def _init_path(self):
        """初始化每台無人機的飛行路徑（圓形路徑為例）"""
        R = 0.3
        PERIOD = 10
        NUM_WP = self.CTRL_FREQ * PERIOD
        self.NUM_WP = NUM_WP
        self.wp_counters = np.array([int((i * NUM_WP / 6) % NUM_WP) for i in range(self.NUM_DRONES)])
        self.target_positions = [[] for _ in range(self.NUM_DRONES)]

        for i in range(NUM_WP):
            for j in range(self.NUM_DRONES):
                x = R * np.cos((i / NUM_WP) * 2 * np.pi + np.pi / 2) + self.INIT_XYZS[j, 0]
                y = R * np.sin((i / NUM_WP) * 2 * np.pi + np.pi / 2) - R + self.INIT_XYZS[j, 1]
                z = self.INIT_XYZS[j, 2] + (i * 0.05 / NUM_WP)

                self.target_positions[j].append(np.array([x, y, z]))

    def _actionSpace(self):
        # 定義每台 drone 的 action 是 9 維（P/I/D for pos, P/I/D for torque）
        # 這樣 RL 模型就可以調整所有控制參數
        act_lower_bound = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array(
            [[1.0, 1.0, 1.0, 0.1, 0.1, 0.1] for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

#
# 📌 建議你先完成 baseline 版本，之後若效果不佳，再考慮 log-scaling。
    def _observationSpace(self):
        # ✅ 每台 drone 的觀測值擴充為 23 維：原始 20 維 + 與目標的 3 維位置差
        obs_lower_bound = np.array([[-np.inf] * 23 for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf] * 23 for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        # ✅ 每個 drone 的觀測值 = 狀態向量 (20) + 目標差 (3)
        obs = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            position = state[:3]
            target_diff = self.target_position[i] - position  # 加入位置差
            obs.append(np.concatenate([state, target_diff]))  # 共 23 維
        return np.array(obs)

    def _preprocessAction(self, action):
        processed = []
        for i in range(self.NUM_DRONES):
            a = action[i]
            # 假設 action 長度為 9：位置 PID (3) + 姿態 PID (3) + thrust limit + yaw rate limit + force limit
            pos_P = 10 ** (3 + 2 * a[0])  # [1e3, 1e5]
            pos_I = 10 ** (-3 + 3 * a[1])  # [1e-3, 1.0]
            pos_D = 10 ** (1 * a[2])  # [1, 10]

            att_P = 0.1 * a[3]  # [0, 0.1]
            att_I = 0.1 * a[4]  # [0, 0.1]
            att_D = 0.1 * a[5]  # [0, 0.1]

            thrust_limit = 60000 + a[6] * 40000  # [60000, 100000]
            yaw_rate_limit = 0 + a[7] * 600  # [0, 600]
            force_limit = 10000 + a[8] * 40000  # [10000, 50000]

            processed.append(np.array([
                pos_P, pos_I, pos_D,
                att_P, att_I, att_D,
                thrust_limit, yaw_rate_limit, force_limit
            ]))
        return np.array(processed)

    def _computeReward(self):
        total_reward = 0.0
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[:3]
            vel = state[10:13]
            target_pos = self.target_position[i]

            pos_error = np.linalg.norm(pos - target_pos)
            vel_penalty = 0.1 * np.linalg.norm(vel)
            upward_reward = 0.5 * pos[2]
            climb_reward = 0.2 if vel[2] > 0 else -0.1

            path_dir = self.target_positions[i][(self.wp_counters[i] + 1) % self.NUM_WP] - pos
            path_dir = path_dir / (np.linalg.norm(path_dir) + 1e-6)
            vel_dir = vel / (np.linalg.norm(vel) + 1e-6)
            align_reward = 0.5 * np.dot(path_dir, vel_dir)

            acc_penalty = 0.0  # 可加速度懲罰（需額外計算）

            reward = -pos_error - vel_penalty + upward_reward + align_reward + climb_reward - 0.1 * acc_penalty
            total_reward += reward

        return total_reward / self.NUM_DRONES

    def _computeTerminated(self):
        # 若 drone 離目標太遠或掉到地面下就觸發結束
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[:3]
            target = self.target_position[i]
            dist = np.linalg.norm(pos - target)
            if dist > 3.0 or pos[2] < 0.05:
                print(f"[TERMINATE] Drone {i} crashed or flew too far: z={pos[2]:.2f}, dist={dist:.2f}")
                return True  # 提早終止 episode
            if self.step_counter > self.CTRL_FREQ * 5:  # 過 baseline 階段再檢查 stuck
                if np.linalg.norm(state[10:13]) < 0.01:
                    self.stuck_counter[i] += 1
                else:
                    self.stuck_counter[i] = 0
                if self.stuck_counter[i] > 100:
                    print(f"[STUCK] Drone {i} seems to be stuck.")
                    return True

        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        # ✅ 回傳每台 drone 的位置誤差（距離目標）
        errors = np.linalg.norm(
            np.array([self._getDroneStateVector(i)[:3] for i in range(self.NUM_DRONES)]) - self.target_position,
            axis=1
        )
        return {"position_error": errors.tolist()}
    def render(self, mode="human"):
        # ✅ 若使用 GUI，將誤差資訊直接畫在畫面上（利用 debug text）
        #如果你之後要顯示更多自訂資訊或軌跡圖等，建議你補上自己的 render() 方法
        if self.GUI:
            for i in range(self.NUM_DRONES):
                current_pos = self._getDroneStateVector(i)[:3]
                error = np.linalg.norm(current_pos - self.target_position[i])
                self.CLIENT_ID = self.CLIENT_ID if hasattr(self, "CLIENT_ID") else 0  # 保險
                import pybullet as p
                p.addUserDebugText(
                    text=f"Drone {i} Error: {error:.2f} m",
                    textPosition=current_pos + np.array([0, 0, 0.3]),
                    textColorRGB=[1, 0, 0],
                    lifeTime=0.1,
                    parentObjectUniqueId=self.DRONE_IDS[i],
                    parentLinkIndex=-1,
                    physicsClientId=self.CLIENT_ID
                )
                p.addUserDebugText(f"P_FOR:{self.ctrl[i].P_COEFF_FOR[0]:.2f}",
                                   textPosition=current_pos + np.array([0.2, 0, 0.5]),
                                   textColorRGB=[0, 0, 1],
                                   lifeTime=0.1,
                                   parentObjectUniqueId=self.DRONE_IDS[i],
                                   parentLinkIndex=-1,
                                   physicsClientId=self.CLIENT_ID)
                p.addUserDebugText(f"I_FOR:{self.ctrl[i].I_COEFF_FOR[0]:.2f}",
                                   textPosition=current_pos + np.array([0.2, 0, 0.6]),
                                   textColorRGB=[0, 0, 1],
                                   lifeTime=0.1,
                                   parentObjectUniqueId=self.DRONE_IDS[i],
                                   parentLinkIndex=-1,
                                   physicsClientId=self.CLIENT_ID)
                p.addUserDebugText(f"D_FOR:{self.ctrl[i].D_COEFF_FOR[0]:.2f}",
                                   textPosition=current_pos + np.array([0.2, 0, 0.7]),
                                   textColorRGB=[0, 0, 1],
                                   lifeTime=0.1,
                                   parentObjectUniqueId=self.DRONE_IDS[i],
                                   parentLinkIndex=-1,
                                   physicsClientId=self.CLIENT_ID)
                p.addUserDebugText(f"P_TOR:{self.ctrl[i].P_COEFF_TOR[0]:.2f}",
                                   textPosition=current_pos + np.array([0.2, 0, 0.8]),
                                   textColorRGB=[0, 0, 1],
                                   lifeTime=0.1,
                                   parentObjectUniqueId=self.DRONE_IDS[i],
                                   parentLinkIndex=-1,
                                   physicsClientId=self.CLIENT_ID)
                p.addUserDebugText(f"I_TOR:{self.ctrl[i].I_COEFF_TOR[0]:.2f}",
                                   textPosition=current_pos + np.array([0.2, 0, 0.9]),
                                   textColorRGB=[0, 0, 1],
                                   lifeTime=0.1,
                                   parentObjectUniqueId=self.DRONE_IDS[i],
                                   parentLinkIndex=-1,
                                   physicsClientId=self.CLIENT_ID)
                p.addUserDebugText(f"D_TOR:{self.ctrl[i].D_COEFF_TOR[0]:.2f}",
                                   textPosition=current_pos + np.array([0.2, 0, 1.0]),
                                   textColorRGB=[0, 0, 1],
                                   lifeTime=0.1,
                                   parentObjectUniqueId=self.DRONE_IDS[i],
                                   parentLinkIndex=-1,
                                   physicsClientId=self.CLIENT_ID)

    @staticmethod
    def action_to_pid_params(action):
        pos_p = 1.0 + action[0] * 2.0  # [1.0, 3.0]
        pos_i = 0.0 + action[1] * 0.1  # [0.0, 0.1]
        pos_d = 0.1 + action[2] * 1.0  # [0.1, 1.1]

        att_p = 30000 + action[3] * 40000  # [30000, 70000]
        att_i = 0 + action[4] * 500  # [0, 500]
        att_d = 10000 + action[5] * 10000  # [10000, 20000]

        return pos_p, pos_i, pos_d, att_p, att_i, att_d

    def reset(self,
              seed: int = None,
              options: dict = None):


        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        # self._init_path()
        # self.target_position = np.array([self.target_positions[i][self.wp_counters[i]] for i in range(self.NUM_DRONES)])

        return initial_obs, initial_info
    def step(self,
             action
             ):


        if self.step_counter < self.CTRL_FREQ * 5:  # 前 5 秒使用穩定 PID
            if self.step_counter == 0:
                print("[INFO] Using baseline PID for warm-up phase")

            for i in range(self.NUM_DRONES):
                self.ctrl[i].P_COEFF_FOR = np.array([.4, .4, 1.25])
                self.ctrl[i].I_COEFF_FOR = np.array([.05, .05, .05])
                self.ctrl[i].D_COEFF_FOR = np.array([.2, .2, .5])
                self.ctrl[i].P_COEFF_TOR = np.array([70000., 70000., 60000.])
                self.ctrl[i].I_COEFF_TOR = np.array([.0, .0, 500.])
                self.ctrl[i].D_COEFF_TOR = np.array([20000., 20000., 12000.])
        else:
            for i in range(self.NUM_DRONES):
                pos_p, pos_i, pos_d, att_p, att_i, att_d = self.action_to_pid_params(action[i])
                self.ctrl[i].P_COEFF_FOR = np.array([pos_p, pos_p, pos_p])
                self.ctrl[i].I_COEFF_FOR = np.array([pos_i, pos_i, pos_i])
                self.ctrl[i].D_COEFF_FOR = np.array([pos_d, pos_d, pos_d])
                self.ctrl[i].P_COEFF_TOR = np.array([att_p, att_p, att_p])
                self.ctrl[i].I_COEFF_TOR = np.array([att_i, att_i, att_i])
                self.ctrl[i].D_COEFF_TOR = np.array([att_d, att_d, att_d])

        # ❷ 產生視覺輸出（錄影用，RECORD 模式開啟時）
        # 不影響控制邏輯，可略過
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter % self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(
                os.path.join(self.IMG_PATH, "frame_" + str(self.FRAME_NUM) + ".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB,  # ImageType.BW, ImageType.DEP, ImageType.SEG
                                      img_input=self.rgb[i],
                                      path=self.ONBOARD_IMG_PATH + "/drone_" + str(i) + "/",
                                      frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                                      )
        #### Read the GUI's input parameters #######################
        #❸ 若開啟 GUI 且使用者在用滑桿控制，則覆蓋 DDPG 輸出
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        # 使用滑桿輸入 RPM，不使用強化學習
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter % (self.PYB_FREQ / 2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:

            # clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
            #### 你的 DSLPID 控制器控制 RPM 的部分 ###########
            # ❹ 初始化 clipped_action（RPM 輸出）
            clipped_action = np.zeros((self.NUM_DRONES, 4))

            for i in range(self.NUM_DRONES):

                # 取得 drone 狀態

                state = self._getDroneStateVector(i)
                # 在 step() 中替換目標點更新邏輯（改為 if 誤差小才跳下一點）
                # 更新 waypoint
                if self.step_counter < self.CTRL_FREQ * 5:
                    self.wp_counters[i] = (self.wp_counters[i] + 1) % self.NUM_WP
                else:
                    if np.linalg.norm(self.target_position[i] - state[:3]) < 0.05:
                        self.wp_counters[i] = (self.wp_counters[i] + 1) % self.NUM_WP
                    self.target_position[i] = self.target_positions[i][self.wp_counters[i]]

                # ❼ 利用 DSLPID 控制器計算對應的 RPM
                rpm, _, _ = self.ctrl[i].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=self.target_position[i],
                    target_rpy=np.zeros(3),
                    target_vel=np.zeros(3),
                )
                print("rpm:",rpm)
                clipped_action[i, :] = rpm
        # ❽ 執行實際模擬步進，更新 PyBullet 狀態
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG,
                                                                Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            for i in range(self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        # ❾ 更新模擬資訊、計算回傳值
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info

