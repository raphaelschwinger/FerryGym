import datetime
import math
import gym
import pygame
import numpy as np
from gym.utils.play import play
from gym.envs.registration import register


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "FerryGymEnv"))

from FerryGymEnv.FerryGymEnv import FerryGymEnv

kwargs = dict(
    generate_training_data=False,
    data_directory='/Users/raphael/git/uni/masterarbeit/ferry-gym/data/rev-moenk/',
    df_filename='2022-04-10-13->14II.pkl',
    startingTime=datetime.datetime(2022, 4, 10, 13, 0, 1),
    
)

gym.register(
    id="FerryGym-v0",
    entry_point="FerryGymEnv.FerryGymEnv:FerryGymEnv",
    kwargs=kwargs,
)


"""
hardcoded policy
this policy turns right a couple of timesteps to avoid hitting ground
and then goes straight to the target
"""

env = gym.make('FerryGym-v0')


def direct_policy(obs):
    target_position = obs['target']
    agent_position = obs['agent_position']
    agent_direction = obs['agent_direction']
    # calculate angle in degrees between target and agent position
    angle = math.degrees(-math.atan2(target_position[1] - agent_position[1],
                                     agent_position[0] - target_position[0])) - 90
    # unit_vector_1 = agent_position / np.linalg.norm(agent_position)
    # unit_vector_2 = target_position / np.linalg.norm(target_position)
    # dot_product = np.dot(unit_vector_1, unit_vector_2)
    # angle = math.degrees(np.arccos(dot_product))
    print(angle)
    a = 2
    b = angle - agent_direction
    return a, b

def hardcoded_policy(obs, step):
    if step > 30:
        return direct_policy(obs)
    else:
        return 2, 1
   


totals = []
for episode in range(10):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        # uncomment next line if you dont want so see the visualization
        # print(obs)
        action = hardcoded_policy(obs, step)
        print("Policy action: ", action)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
        env.render()
    totals.append(episode_rewards)

print(totals)
