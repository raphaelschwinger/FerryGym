# add plot
from turtle import width
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import os


# local modules
from FerryGymEnv import FerryGymEnv
from MapRevMoenk import BOUND_RIGHT, BOUND_LEFT, BOUND_TOP, BOUND_BOTTOM

env = FerryGymEnv()
max_x, max_y = env.convertLatLotInEnvCoordinates(BOUND_BOTTOM, BOUND_RIGHT)
axis_measure = [0, max_x, 0, max_y]

def plot_trajectories(trajectories: Tuple[List[Dict[float, float]], str]):
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "15-50-rev-moenk.png")
    img = plt.imread(img_path)
    plt.imshow(img, extent=axis_measure, origin='lower')
    plt.gca().invert_yaxis()
    # for every mmsi in df plot the trajectory
    for trajectory, label in trajectories:
        # distinct color for mmsi
        color = np.random.rand(3,)
        # plot line
        plt.plot([x for x, y in trajectory], [y for x, y in trajectory], label=label, color=color)
    # plot target position [2244.34601982, 90.73506014]
    plt.plot(2244.34601982, 90.73506014, label='target', color='red', marker='o')
    plt.legend()

def plot_states(states: Tuple[List[Dict[float, float]], str, str], margin=500):
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "15-50-rev-moenk.png")
    img = plt.imread(img_path)
    plt.imshow(img, extent=axis_measure, origin='lower')
    plt.gca().invert_yaxis()
    # for every mmsi in df plot the trajectory
    for trajectory, label, color in states:
        # distinct color for mmsi
        part_trajectory = trajectory[::5]
        plt.quiver([x for x, y, _, _ in part_trajectory], [y for x, y, _, _ in part_trajectory], 1,1, angles=[90 - d for _, _, _, d in part_trajectory], width=0.003, pivot='middle', color=color, label=label)
        # plot line
        plt.plot([x for x, y, _, _ in trajectory], [y for x, y, _, _ in trajectory], color=color) 
    # get min x value
    min_x = min([min([x for x, y, _, _ in trajectory]) for trajectory, _, _ in states])
    # get max x value
    max_x = max([max([x for x, y, _, _ in trajectory]) for trajectory, _, _ in states])
    # get min y value
    min_y = min([min([y for x, y, _, _ in trajectory]) for trajectory, _, _ in states])
    # get max y value
    max_y = max([max([y for x, y, _, _ in trajectory]) for trajectory, _, _ in states])
    plt.xlim(min_x - margin, max_x + margin)
    plt.ylim(max_y + margin, max(min_y - margin, 0))
    plt.legend()

def plot_df(df, margin = 500):
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "15-50-rev-moenk.png")
    img = plt.imread(img_path)
    plt.imshow(img, extent=axis_measure, origin='lower')
    plt.gca().invert_yaxis()

    # for every mmsi in df plot the trajectory
    for mmsi in df['mmsi'].unique():
        # distinct color for mmsi
        color = np.random.rand(3,)
        # iterate though rows for mmsi starting with smallest time
        for index, row in df[df['mmsi'] == mmsi].sort_values(by=['datetime']).iterrows():
            # plot dashed line
            plt.plot(row.x, row.y, color=color, marker='o', linestyle='dashed',
                linewidth=1, markersize=2)
    # get min x value
    min_x = min(df['x'])
    # get max x value
    max_x = max(df['x'])
    # get min y value
    min_y = min(df['y'])
    # get max y value
    max_y = max(df['y'])
    plt.xlim(min_x - margin, max_x + margin)
    plt.ylim(max_y + margin, max(min_y - margin, 0))

def heatmap2d(arr: np.ndarray):
    
    plt.imshow(arr, cmap='viridis', extent=axis_measure, origin='lower')
    # plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()

