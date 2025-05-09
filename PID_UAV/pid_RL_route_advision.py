"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories
in the X-Y plane, around point (0, -.3).
pid_RL_route_advision.py
"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.PID_RL import PID_RL#修正
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

prev_position = [[0, 0, 0] for _ in range(DEFAULT_NUM_DRONES)]
TARGET_position = []





def train_ddpg_model(env, model_path, total_iterations, timesteps_per_iter):
    rewards_log = []

    # 取出內部實際的 PID_RL 環境
    base_env = env.env

    # 建立新的模型
    model = DDPG("MlpPolicy", env, verbose=1)
    print("[INFO] Created a new DDPG model from scratch.")

    for i in range(total_iterations):
        print(f"\n[TRAINING] Iteration {i + 1}/{total_iterations}")

        # 建立評估環境（重新初始化一個環境實例，參數從 base_env 取得）
        eval_env = Monitor(PID_RL(
            drone_model=base_env.DRONE_MODEL,
            num_drones=base_env.NUM_DRONES,
            initial_xyzs=base_env.INIT_XYZS,
            initial_rpys=base_env.INIT_RPYS,
            physics=base_env.PHYSICS,
            neighbourhood_radius=10,
            pyb_freq=base_env.PYB_FREQ,
            ctrl_freq=base_env.CTRL_FREQ,
            gui=False,
            record=False,
            obstacles=base_env.OBSTACLES,
            user_debug_gui=False
        ))

        # 建立 callback
        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=os.path.dirname(model_path),
                                     log_path=os.path.dirname(model_path),
                                     eval_freq=500,
                                     deterministic=True,
                                     render=False,
                                     verbose=1)

        # 訓練
        model.learn(total_timesteps=timesteps_per_iter,
                    reset_num_timesteps=False,
                    callback=eval_callback)

        # 儲存中間模型
        iter_model_path = f"{model_path}_iter{i + 1}"
        model.save(iter_model_path)
        print(f"[SAVED] Model saved to {iter_model_path}.zip")

        # 評估與記錄 reward
        reward = evaluate_model(eval_env, model)

        rewards_log.append(reward)
        print(f"[EVAL] Reward: {reward:.2f}")

    # 儲存最終模型
    model.save(model_path)
    print(f"[DONE] Final model saved to {model_path}.zip")

    # 畫訓練 reward 曲線
    try:
        plt.plot(rewards_log)
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("DDPG Training Reward")
        plt.grid()
        reward_plot_path = os.path.join(os.path.dirname(model_path), "training_rewards.png")
        plt.savefig(reward_plot_path)
        print(f"[PLOT] Saved reward curve to {reward_plot_path}")
        plt.close()
    except Exception as e:
        print("[WARN] Could not plot:", e)



def evaluate_model(env, model, num_episodes=1):
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for _ in range(1000):  # Max steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)




def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################

    H = .1
    H_STEP = .05
    R = .3
    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    INIT_XYZS = np.array([
        [R * np.cos((i / 6) * 2 * np.pi + np.pi / 2),
         R * np.sin((i / 6) * 2 * np.pi + np.pi / 2) - R,
         H + i * H_STEP]
        for i in range(num_drones)
    ])
    INIT_RPYS = np.array([[0, 0, i * (np.pi / 2) / num_drones] for i in range(num_drones)])

    for j in range(num_drones): #紀錄起始位置
        prev_position[j] = INIT_XYZS[j]
    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = [
            R * np.cos((i / NUM_WP) * 2 * np.pi + np.pi / 2) + INIT_XYZS[0, 0],
            R * np.sin((i / NUM_WP) * 2 * np.pi + np.pi / 2) - R + INIT_XYZS[0, 1],
            0
        ]
        TARGET_position.append(TARGET_POS[i, :])

    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])


    #### Create the environment ################################
    env = Monitor(PID_RL(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        ))
    # model = DDPG("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=50)
    # >>>> [1] 訓練模型（訓練完一次，產生模型） <<<<
    MODEL_PATH = os.path.join(output_folder, "ddpg_model")
    TOTAL_TRAIN_ITER = 5
    TIMESTEPS_PER_ITER = 50
    train_ddpg_model(env, MODEL_PATH, total_iterations=TOTAL_TRAIN_ITER, timesteps_per_iter=TIMESTEPS_PER_ITER)
    model = DDPG.load(MODEL_PATH, env=env)
    # model = train_ddpg_model(env, MODEL_PATH, total_iterations=TOTAL_TRAIN_ITER, timesteps_per_iter=TIMESTEPS_PER_ITER)

    # >>>> [2] 重設環境，使用訓練好的模型進行測試 <<<<
    obs, _ = env.reset()

    #### Obtain the PyBullet Client ID from the environment ####
    # PYB_CLIENT = env.getPyBulletClient()
    PYB_CLIENT = env.env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )



    #### Run the simulation ####################################


    # action = np.array([[0.5, 0.5, 0.5,  # P_COEFF_FOR
    #                     0.01, 0.01, 0.01,  # I_COEFF_FOR
    #                     80000, 300.0, 30000] for _ in range(num_drones)], dtype=np.float32)  # Torque PID

    # action = np.array([[1.0, 1.0, 1.0] for _ in range(num_drones)])  # 初始 PID 參數（可由 RL 調整）
    START = time.time()
    for i in range(0, int(duration_sec * env.env.CTRL_FREQ)):

        # 每次更新目標位置傳入環境
        # step_target_positions = np.vstack([
        #     np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2] + (i * H_STEP / NUM_WP)])
        #     for j in range(num_drones)
        # ])
        #
        # env.set_target_position(step_target_positions)
        # model = train_ddpg_model(env, MODEL_PATH, TOTAL_TRAIN_ITER,TIMESTEPS_PER_ITER)

        # action, _ = model.predict(obs, deterministic=True)
        action, _states = model.predict(obs, deterministic=True)
        #### Step 環境讓他自己控制 ##############################
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        #### Go to the next way point and loop #####################
        # for j in range(num_drones):
        #     wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            try:
                logger.log(drone=j,
                           timestamp=i / env.env.CTRL_FREQ,
                           state=env.env._getDroneStateVector(0),
                           control=np.hstack(
                               [TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)]),

                           ) # 或你實際的控制項

            except Exception as e:
                print(f"[ERROR] Logger.log() failed at step {i}: {e}")
            # logger.log(drone=j,
            #            timestamp=i / env.CTRL_FREQ,
            #            state=obs[j],
            #            control=np.hstack(
            #                [TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)]))

        #### Printout ##############################################
        # env.render()

        #### Sync the simulation ###################################
        if gui:
            env.render()
            sync(i, START, env.env.CTRL_TIMESTEP)

    ###########################################################
    #### Initialize the controllers ############################
    # if drone in [DroneModel.CF2X, DroneModel.CF2P]:
    #     ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]
    # action = np.zeros((num_drones,4))
    # START = time.time()
    # for i in range(0, int(duration_sec*env.CTRL_FREQ)):
    #
    #     #### Make it rain rubber ducks #############################
    #     # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)
    #
    #     #### Step the simulation ###################################
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print("obs=",obs)
    #     #### Compute control for the current way point #############
    #     for j in range(num_drones):
    #         action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
    #                                                                 state=obs[j],
    #                                                                 target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
    #                                                                 # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
    #                                                                 target_rpy=INIT_RPYS[j, :]
    #                                                                 )
    #         # print("target_pos=",TARGET_position)
    #
    #     #### Go to the next way point and loop #####################
    #     for j in range(num_drones):
    #         wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0
    #
    #     #### Log the simulation ####################################
    #     for j in range(num_drones):
    #         logger.log(drone=j,
    #                    timestamp=i/env.CTRL_FREQ,
    #                    state=obs[j],
    #                    control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
    #                    # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
    #                    )
    #
    #     #### Printout ##############################################
    #     env.render()
    #
    #     #### Sync the simulation ###################################
    #     if gui:
    #         sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save
    #在 Logger 的記錄中加入 reward 觀察學習情況(未做!!!)

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()
        for i in range(NUM_WP):
            TARGET_POS[i, :] = [
                R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2),  # X
                R * np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2) - R,  # Y
                H  # Z
            ]

        # 繪製 3D 圖形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(TARGET_POS[:, 0], TARGET_POS[:, 1], TARGET_POS[:, 2], label="理想軌跡", color='r')

        # 標示軸
        ax.set_xlabel("X ")
        ax.set_ylabel("Y ")
        ax.set_zlabel("Z ")
        ax.legend()
        plt.show()

if __name__ == "__main__":
    # for episode in range(100):
    #     print(f"--- Episode {episode} ---")
    #     parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    #     parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, help='Drone model (default: CF2X)',
    #                         metavar='', choices=DroneModel)
    #     parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int, help='Number of drones (default: 3)',
    #                         metavar='')
    #     parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, help='Physics updates (default: PYB)',
    #                         metavar='', choices=Physics)
    #     parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
    #                         help='Whether to use PyBullet GUI (default: True)', metavar='')
    #     parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
    #                         help='Whether to record a video (default: False)', metavar='')
    #     parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
    #                         help='Whether to plot the simulation results (default: True)', metavar='')
    #     parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
    #                         help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    #     parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
    #                         help='Whether to add obstacles to the environment (default: True)', metavar='')
    #     parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
    #                         help='Simulation frequency in Hz (default: 240)', metavar='')
    #     parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
    #                         help='Control frequency in Hz (default: 48)', metavar='')
    #     parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
    #                         help='Duration of the simulation in seconds (default: 5)', metavar='')
    #     parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
    #                         help='Folder where to save logs (default: "results")', metavar='')
    #     parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
    #                         help='Whether example is being run by a notebook (default: "False")', metavar='')
    #     ARGS = parser.parse_args()
    #
    #     run(**vars(ARGS))

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

