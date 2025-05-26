import numpy as np 
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gym_pybullet_drones.envs.pid_RL import pid_RL

duration_sec    = 12      
control_freq_hz = 48      
total_steps     = duration_sec * control_freq_hz 
period_sec      = 2.0     
R               = 0.3
INIT_Z          = 0.1
UP_SPEED        = 0.07
num_drones      = 1

# ── 2) 計算 INIT_XYZS, INIT_RPYS（和 pid.py 一樣）
INIT_XYZS = np.array([[
    R*np.cos((i/6)*2*np.pi + np.pi/2),
    R*np.sin((i/6)*2*np.pi + np.pi/2) - R,
    INIT_Z + i*0.05
] for i in range(num_drones)])
INIT_RPYS = np.array([[0, 0, i*(np.pi/2)/num_drones]
                      for i in range(num_drones)])

# ── 3) 生成理想螺旋軌跡
t     = np.arange(total_steps) / control_freq_hz        
angle = 2*np.pi * t / period_sec + np.pi/2        
x     = R * np.cos(angle)
y     = R * np.sin(angle) - R
z     = INIT_Z + UP_SPEED * t
ideal_traj = np.stack([x, y, z], axis=1)  

# ── 4) 包裝環境
def make_env():
    env = pid_RL(
        num_drones   = num_drones,
        gui          = True,
        target_traj  = ideal_traj,
        max_steps    = total_steps,
        initial_xyzs = INIT_XYZS,
        initial_rpys = INIT_RPYS,
    )
    return Monitor(env)

env = DummyVecEnv([make_env])

eval_callback = EvalCallback(
    env,
    best_model_save_path='./logs/best/',
    log_path='./logs/eval/',
    eval_freq=5000,
    deterministic=True,
    render=False
)
model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=100_000,
    verbose=1
)
#tensorboard
model.learn(
    total_timesteps=200_000,
    log_interval=10,
    tb_log_name="pid_run",
    callback=eval_callback
)

# ── 6) 抓回最終的 PID 參數
single_env = env.envs[0]
obs, _     = single_env.reset()
delta, _   = model.predict(obs, deterministic=True)
pid_params = single_env.base_pid + delta[0]
P, I, D    = pid_params[:3], pid_params[3:6], pid_params[6:9]

print("訓練後 PID 參數：")
print("P_coeff:", P)
print("I_coeff:", I)
print("D_coeff:", D)

model.save("sac_pid_helix")
print("這次一定行！")
