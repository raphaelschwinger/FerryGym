import math
import gym
import pygame
import numpy as np
from gym.utils.play import play


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "FerryGymEnv"))

from FerryGymEnv import FerryGymEnv

env = gym.make('FerryGym-v0')


def basic_policy(obs):
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


totals = []
for episode in range(10):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        # uncomment next line if you dont want so see the visualization
        print(obs)
        action = basic_policy(obs)
        print("Policy action: ", action)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
        env.render()
    totals.append(episode_rewards)

print(totals)
