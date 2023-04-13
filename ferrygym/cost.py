import numpy as np
import torch
import math

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1 / (2*math.pi*sx*sy) * \
      torch.exp(-((x - mx)**2 / (2*sx**2) + (y - my)**2 / (2*sy**2)))

def get_proximity_mask(size):
    """
    Generate a proximity mask for the ferrygym environment.
    The mask is used to compute the cost of the agent's actions for the PPUU algorithm.

    Parameters
    ----------
    size : int
        The size of the proximity mask.

    Returns
    -------
    torch.Tensor
        The proximity mask, represented as a 2D tensor.
    """
    # create tensor of size NEIGHBORHOOD_SIZE x NEIGHBORHOOD_SIZE
    mask = torch.zeros((size, size))
    # add value of 1 to the center of the mask of size 1x4
    mask[48:52, 49:50] = 1
    # add gaussian distribution to the middle of the mask
    x = torch.linspace(0, mask.shape[0], mask.shape[0])
    y = torch.linspace(0, mask.shape[1], mask.shape[1])
    x, y = torch.meshgrid(x, y)
    gaussian_mask = torch.clamp(100* gaussian_2d(x, y, mx=49.5, my=49.5, sx=5, sy=10), 0, 1)

    # TODO: different probabilities for front and back
    # top_mask = torch.empty(FerryGymEnv.constants.NEIGHBORHOOD_SIZE, FerryGymEnv.constants.NEIGHBORHOOD_SIZE)
    # # set bottom half of the mask to 0
    # top_mask[:int(FerryGymEnv.constants.NEIGHBORHOOD_SIZE/2), :] = gaussian_mask[int(FerryGymEnv.constants.NEIGHBORHOOD_SIZE/2):, :]
    # # set top half of the mask to the gaussian distribution
    # top_mask[int(FerryGymEnv.constants.NEIGHBORHOOD_SIZE/2):, :] = 0 

    return torch.max(mask, gaussian_mask)

def cost_function(observation, proximity_mask):
    # target cost
    distance_to_target = np.linalg.norm(np.array(observation['agent_position']) - np.array(observation['target']))
    max_distance = np.linalg.norm(np.array(observation['agent_starting_position']) - np.array(observation['target']))
    target_cost = distance_to_target / max_distance
    # proximity cost to other ships
    proximity_cost = torch.max(proximity_mask * observation['neighborhood'][:,:,1]).numpy()
    proximity_cost = proximity_cost / 255
    # proximity cost to land
    land_cost = torch.max(proximity_mask * observation['neighborhood'][:,:,0]).numpy()
    land_cost = land_cost / 255
    return target_cost, proximity_cost, land_cost
