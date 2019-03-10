#!/usr/bin/env python3

from obstacle_tower_env import ObstacleTowerEnv
from matplotlib import pyplot as plt 

ENV_PATH = './obstacle-tower-challenge/ObstacleTower/obstacletower'
env = ObstacleTowerEnv(ENV_PATH, retro=False, realtime_mode=True)

# Seeds can be chosen from range of 0-100.
env.seed(5)

# Floors can be chosen from range of 0-24.
env.floor(15)

# The environment provided has a MultiDiscrete action space, where the 4 dimensions are:

# 0. Movement (No-Op/Forward/Back)
# 1. Camera Rotation (No-Op/Counter-Clockwise/Clockwise)
# 2. Jump (No-Op/Jump)
# 3. Movement (No-Op/Right/Left)

print('action space', env.action_space)

# The observation space provided includes a 168x168 image (the camera from the simulation)
# as well as the number of keys held by the agent (0-5) and the amount of time remaining.

print('observation space', env.observation_space)

# Interacting with the environment

obs = env.reset()
plt.imshow(obs[0])

# Get action meanings
print('Table of actions')
for action_id, action_meaning in enumerate(env.get_action_meanings()):
    print(action_id, action_meaning)

import signal

def env_closer(signo, handler):
    print('Closing the environment...')
    import sys; sys.exit(1)
    env.close()

signal.signal(signal.SIGINT, env_closer)

while True:
    sampled_action = env.action_space.sample()
    print('Sampled action:', sampled_action)
    
    obs, reward, done, info = env.step(sampled_action)
    plt.imshow(obs[0])
    print('Reward after action', reward)
