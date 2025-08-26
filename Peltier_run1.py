
from Peltier_Env1 import PeltierEnv
import gymnasium as gym
import os
import time
from stable_baselines3 import PPO

model_dir ="model/PPO_Peltier1"
logdir = "logs_Peltier1"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = PeltierEnv()
env.reset()

model1 = PPO("MlpPolicy", env, verbose= 1, tensorboard_log= logdir, n_steps=2500, batch_size=128, device='cuda',gamma=0.99)
TIMESTEPS = 500

for i in range(1,10000):
    model1.learn(total_timesteps= TIMESTEPS, reset_num_timesteps= False, tb_log_name="PPO")
    model1.save(f"{model_dir}/{TIMESTEPS*i}")

env.close()