import gym
import pygame
from gym.utils.play import play
import numpy as np

from gym.envs.registration import register
from FerryGymEnv import FerryGymEnv


kwargs = dict(
    generate_training_data=False,
    other_ships=True,
)

register(
    id="FerryGym-v0",
    entry_point="FerryGymEnv:FerryGymEnv",
    kwargs=kwargs,
)
mapping = {(pygame.K_LEFT,): np.array([0,-5]) , (pygame.K_RIGHT,): np.array([0,5]),
           (pygame.K_UP,): np.array([2,0]), (pygame.K_DOWN,): np.array([-2,0])}

env = gym.make('FerryGym-v0')

play(env, keys_to_action=mapping, noop=np.array([0,0]))
