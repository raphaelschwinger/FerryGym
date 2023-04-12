import datetime
import os
import gym
from gym import spaces
from gym.utils.renderer import Renderer
import pygame
import numpy as np
import pandas as pd
import json
import matplotlib.path as mpltPath
import sys
import copy


# add local path to sys.path to import local modules 
sys.path.append((os.path.dirname(__file__)))
print('path ferrygym', sys.path)

# local imports
from utils.CoordinateTransform import LatLonToPxPy, MetersToPxPy, PxPyToMeters, get_scale

from Ship import Ship
from constants import *
from MapRevMoenk import *
from utils.graphics import draw_polygone, draw_rect, rotate_points
from cost import get_proximity_mask, cost_function

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

WINDOW_SIZE_WIDTH = 1135
WINDOW_SIZE_HEIGHT = 1084

class FerryGymEnv(gym.Env):
    """
    ## Description

    This environment should simulate a ferry agent navigating in the Kiel fjord.
    The goal is to reach a target location while staying on the fjord and avoiding collisions with other ships.

    ## Action space

    The ferry can accelerate (or brake) and turn its heading.
    Acceleration is measured in m/s^2, and and limited to physical possible values.
    Turning is measured in degree.

    ## Observation space
        
    The observation space is a dictionary containing the following keys:
        - `agent_position`: A 2D vector representing the position of the agent in environment coordinates.
        - `agent_direction`: A floating point value representing the direction the agent is facing in degrees.
        - `agent_speed`: A floating point value representing the speed of the agent.
        - `target`: A 2D vector representing the position of the target in latitude and longitude coordinates.
        - `neighborhood`: A 3D array of unsigned integers with a shape of (NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE, 3) representing the agent's local environment.
        - `agent_starting_position`: A 2D vector representing the starting position of the agent in latitude and longitude coordinates.

    NOTE: This environment can be used to generate the training set for the [PPUU](https://github.com/Atcold/pytorch-PPUU) algorithm. Use the 'generate_training_data' flag to enable this mode. This will save a new pandas dataset with an additional column containing the path to a file containing the neighborhood for each ship and timestep.



    ## Episode termination

    The episode ends if one of the following occurs:
     1. Termination: the agent reaches the target.
     2. The agent is out of the fjord.
     3. The agent collides with another ship.
     4. If a dataset of ship trajectories is provided, the dataset is exhausted.

    ## Rewards
    The agent receives the following rewards:
        - running: if nothing happens, the agent receives a default reward of 0.
        - dataset_exhausted: if the provided dataset is exhausted, the agent receives a default reward of 0.
        - success: if the agent reaches the target, it receives a default reward of 100.
        - collision: if the agent collides with another ship, it receives a default reward of -100.
        - out_of_bounds: if the agent leaves the fjord, it receives a default reward of -100.

        
    ### Start and target position

    The starting state is the ferry dock Reventlue (N54°19.961' E10°09.190 | 54.332683, 10.153167), the target is the ferry dock Moenkeberg (N54°21.117' E10°10.587' | 54.351950, 10.176450).
    """
 
    metadata = {"render_modes": [
        "human", "rgb_array", "single_rgb_array"], "render_fps": 5}
    
    MAX_DISTANCE_TO_TARGET = 20.0
    TIME_STEP = datetime.timedelta(seconds=1)

    def __init__(
        self,
        render_mode=None,
        generate_training_data=False,
        other_ships=False,
        data_directory='./ferrygym/',
        df_filename='dataset.pkl',
        render_observation=True,
        rewards={
            'running': 0,
            'collision': -100,
            'dataset_exhausted': 0,
            'success': 100,
            'out_of_bounds': -100,
            },
        ):
        """
        Initialize the environment for the ferrygym.

        Parameters
        ----------
        render_mode : str, optional
            The rendering mode to use for the environment. The default value is None.
        generate_training_data : bool, optional
            Whether to generate training data. The default value is False.
        other_ships : bool, optional
            Whether to include other ships in the environment. The default value is False.
        data_directory : str, optional
            The directory where the training data is stored. The default value is './ferrygym/'.
        df_filename : str, optional
            The filename of the dataset. The default value is 'dataset.pkl'.
        render_observation : bool, optional
            Whether to render the observation. The default value is True.
        rewards : dict, optional
            The rewards for different actions in the environment. The default value is:
            {
                'running': 0,
                'collision': -100,
                'dataset_exhausted': 0,
                'success': 100,
                'out_of_bounds': -100,
            }

        Returns
        -------
        None
        """
        self.rewards = rewards
        self.generate_training_data = generate_training_data
        self.other_ships = other_ships
        self.data_directory = data_directory
        self.df_filename = df_filename
        self.render_observation = render_observation
        if not self.render_observation:
            '''
            If the observation is not rendered, the observation space is a 3D array of zeros.       
            '''
            self.zero_observation = np.zeros((NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE, 3), dtype=np.uint8) 

        self.timestep = 0 # counter of steps in the current episode
        self.dt = 1.0  # time step in seconds

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as a latitude and longitude coordinate.
        self.observation_space = spaces.Dict(
            {
                "agent_position": spaces.Box(low=np.array([54.000, 10.00, 0.0]), high=np.array([54.500, 10.500, MAX_VELOCITY]), dtype=np.float32),
                "agent_direction": spaces.Box(low=0.0, high=360.0, dtype=np.float32),
                "agent_speed": spaces.Box(low=0.0, high=MAX_VELOCITY, dtype=np.float32),
                "target": spaces.Box(low=np.array([54.000, 10.00]), high=np.array([54.500, 10.500]), dtype=np.float32),
                "neighborhood": spaces.Box(low=0, high=255, shape=(NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE, 3), dtype=np.uint8),
                "agent_starting_position": spaces.Box(low=np.array([54.000, 10.00, 0.0]), high=np.array([54.500, 10.500, MAX_VELOCITY]), dtype=np.float32),
            }
        )

        # a action is a vector of length 2, containing the acceleration and the turning angle
        self.action_space = spaces.Box(low=np.array(
            [MIN_ACCELERATION, - MAX_DIRECTION]), high=np.array([MAX_ACCElERATION, MAX_DIRECTION]), dtype=np.float32)

        self.proximity_mask = get_proximity_mask(NEIGHBORHOOD_SIZE)

        '''
        Initialize variables, populated in reset()
        '''
        self.agent = None
        self.agent_starting_position = None
        self.ships = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.background = pygame.image.load(os.path.join(__location__,'15-50-rev-moenk.png'))

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.clock = None
        self.window = None

        """
        Variables specific to rendering the 15-rev-moenk map
        """
        self.sea_land_grid = self._load_sea_land_grid()

        """
        Load df containing AIS data of other ships
        """
        if (self.other_ships):
            self.df = pd.read_pickle(self.data_directory + self.df_filename)
            self.df_datetime = self.df.groupby('datetime')
            # set current_datetime to first time in df
            self.startingTime = self.df_datetime.first().index[0].to_pydatetime()
            self.current_datetime = self.df_datetime.first().index[0].to_pydatetime()
        self.min_distance_to_other_ships = None

    def reset(self, seed=None, return_info=False, options={
        'startPosition': [54.332833, 10.155],
        'startDirection': 35.2,
        'startSpeed': 0.0,
        'startingTime': None,
        'random_startingTime': False,
        }):
        """
        Reset the environment to its initial state.

        This function resets the environment, the agent, and other variables to their initial states.
        It can be used to restart the environment with new starting conditions specified by the options dictionary.

        Parameters:
            seed (int, optional): Seed for random number generator. Default is None.
            return_info (bool, optional): If True, return additional info. Default is False.
            options (dict, optional): Dictionary containing the options for resetting the environment.
                startPosition (list, optional): [latitude, longitude] of the agent's start position. Default is [54.332833, 10.155].
                startDirection (float, optional): Initial direction (in degrees) of the agent. Default is 35.2.
                startSpeed (float, optional): Initial speed of the agent. Default is 0.0.
                startingTime (datetime, optional): Starting time for the simulation. Default is None.
                random_startingTime (bool, optional): If True, use a random starting time. Default is False.

        Returns:
            tuple: A tuple containing the following elements:
                - observation: The initial observation after resetting the environment.
                - info (optional): Additional information about the environment's state.
                  Returned only if return_info is set to True.
        """
        self.timestep = 0 # reset step counter 
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        print('reset called')
        if options['random_startingTime']:
            # get random row in df
            datetime = self.df.sample(1).iloc[0]['datetime'].to_pydatetime()
            self.current_datetime = datetime
        else:
            if options['startingTime'] is None:
                if self.other_ships:
                    # set current_datetime to first time in df
                    self.current_datetime = self.df_datetime.first().index[0].to_pydatetime()
                    self.current_datetime = self.startingTime
                else:
                    self.current_datetime = self.startingTime
            else:
                self.current_datetime = options['startingTime']

        self.min_distance_to_other_ships = None

        self.agent = Ship(
            position=self.convertLatLotInEnvCoordinates(
                lat=options['startPosition'][0], lon=options['startPosition'][1]),
            direction=options['startDirection'],
            dt=self.dt,
            velocity=options['startSpeed'],
        )
        self.agent_starting_position = copy.copy(self.agent.position)

        # set the target location to Moenkeberg
        self._target_location = self.convertLatLotInEnvCoordinates(
            lat=54.351917, lon=10.176400)

        observation = self._get_obs()
        info = self._get_info()

        self.renderer.reset()
        self.renderer.render_step()

        return (observation, info) if return_info else observation

    def step(self, action):
        """
        Advance the environment by one time step based on the provided action.

        This function updates the environment and agent state based on the given action. It also checks
        for various terminal conditions, such as reaching the target, going out of bounds, or colliding
        with another ship. If the generate_training_data flag is set, it generates training data
        by rendering the neighborhood images for all ships.

        Parameters:
            action (tuple): Action taken by the agent in the environment.

        Returns:
            tuple: A tuple containing the following elements:
                - observation: The current observation of the environment after the action is taken.
                - reward (float): Reward received by the agent for taking the action.
                - done (bool): True if the episode has reached a terminal state, False otherwise.
                - info (dict): Additional information about the environment's state.
        """
        if self.other_ships:
            # check if current_time is in df
            if not self.current_datetime.strftime('%Y-%m-%dT%H:%M:%S.000Z') in self.df['datetime'].unique():
                print('stopped because current time is not in dataset: ', self.current_datetime)
                done = True
                reward = self.rewards['dataset_exhausted']
                info = self._get_info('dataset ended')
                observation = self._get_obs()
                return (observation, reward, done, info)
            # load ships for current time
            self._get_ships()
        if self.generate_training_data:
            self._generate_training_data()
            done = False
            reward = self.rewards['running']
        else:
            self.agent.step(action)
            done, reward, info = self._check_agent_conditions()

        observation = self._get_obs()
        self.renderer.render_step()
        self.timestep += 1
        self.current_datetime += self.TIME_STEP
        return observation, reward, done, info

    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def close(self):
        """
        Closes the environment and saves the dataframe to a file if generate_training_data is True.

        This method saves the dataframe to a file with a specific filename in the data_directory
        if generate_training_data is True. The filename is generated based on the starting time.
        It also closes the pygame window if it is open.
        """
        if self.generate_training_data:
            filename = self.data_directory + 'df_' + str(self.startingTime) + '.pkl'
            print('saving df: ', filename)
            pd.to_pickle(self.df, filename)
        
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        """
        Get the observation for the ferrygym environment.

        Returns
        -------
        dict
            The observation, represented as a dictionary with the following keys:
            - "agent_position": the position of the agent
            - "agent_direction": the direction of the agent
            - "agent_speed": the speed of the agent
            - "target": the location of the target
            - "neighborhood": the neighborhood around the agent as a , set to zero if render_observation is False
            - "agent_starting_position": the starting position of the agent
        """
        if self.render_observation:
            return {"agent_position": self.agent.position, "agent_direction": self.agent.direction, "agent_speed": self.agent.velocity, "target": self._target_location, "neighborhood": self._render_neighborhood(self.agent), "agent_starting_position": self.agent_starting_position}
        else:
            return {"agent_position": self.agent.position, "agent_direction": self.agent.direction, "agent_speed": self.agent.velocity, "target": self._target_location, "neighborhood": self.zero_observation, "agent_starting_position": self.agent_starting_position}

    def _get_info(self, status="running"):
        """
        Get information about the ferrygym environment.

        Parameters
        ----------
        status : str, optional
            The status of the environment. The default value is "running".
    
        Returns
        -------
        dict
            The information, represented as a dictionary with the following keys:
            - "status": the status of the environment
            - "distance": the minimum distance to other ships
            - "datetime": the current datetime in the environment
        """
        return {
            "status": status,
            "distance": self.min_distance_to_other_ships,
            "datetime": self.current_datetime,
        }

    def _generate_training_data(self):
        """
        Generates training data for each ship in the environment. For each ship, a neighborhood image is rendered,
        saved to disk, and its filename is stored in the associated DataFrame.

        This method assumes that the following attributes exist:
            - self.ships: list of Ship objects in the environment
            - self.current_datetime: current simulation datetime
            - self.data_directory: directory path where the images will be saved
            - self.df: DataFrame containing the ship data
        """
        for ship in self.ships:
            # generate neighborhood images
            neighborhood = self._render_neighborhood(ship)
            # generate unique filename
            filename = 'nbi_' + str(self.current_datetime) + '_' + str(ship.mmsi) + '.npy'
            # save neighborhood image
            np.save(self.data_directory + 'images/' + filename, neighborhood)
            # store filename in dataframe
            self.df.loc[self.df.index[ship.df_index - 1], 'filename'] = filename

    def _check_agent_conditions(self):
        """
        Checks the agent's current conditions and determines whether the agent has reached the target, is out of bounds,
        or has collided with another ship. Updates the reward, done, and info variables accordingly.

        Returns:
            done (bool): True if the agent has reached the target, collided with another ship, or is out of bounds; False otherwise.
            reward (float): The reward based on the agent's current conditions.
            info (dict): A dictionary containing information about the agent's current status.
        """
        done = False
        if self._agent_reached_target():
            print('agent reached the target')
            info = self._get_info(status="success")
            done = True
            reward = self.rewards['success']
        else:
            reward = self.rewards['running']
            info = self._get_info()

        if not done:
            done, info, reward = self._check_agent_out_of_bounds(done, info, reward)
            done, info, reward = self._check_agent_collision(done, info, reward)

        return done, reward, info

    def _agent_reached_target(self):
        return (self.agent.position[0] <= self._target_location[0] + self.MAX_DISTANCE_TO_TARGET and
                self.agent.position[0] >= self._target_location[0] - self.MAX_DISTANCE_TO_TARGET and
                self.agent.position[1] <= self._target_location[1] + self.MAX_DISTANCE_TO_TARGET and
                self.agent.position[1] >= self._target_location[1] - self.MAX_DISTANCE_TO_TARGET)

    def _check_agent_out_of_bounds(self, done, info, reward):
        if not self.seaPolygonPath.contains_points(self.agent.getPathInDirection(), None, 2).all():
            print("agent is outside sea")
            done = True
            info = self._get_info("land")
            reward = self.rewards['out_of_bounds']
        return done, info, reward

    def _check_agent_collision(self, done, info, reward):
        """
        Checks if the agent is colliding with another ship. If a collision is detected, updates the done, info,
        and reward variables accordingly. Also calculates the minimum distance to other ships.

        Args:
            done (bool): The current done status of the agent.
            info (dict): A dictionary containing information about the agent's current status.
            reward (float): The current reward based on the agent's conditions.

        Returns:
            done (bool): Updated done status, True if the agent is colliding with another ship; False otherwise.
            info (dict): Updated dictionary containing information about the agent's current status.
            reward (float): Updated reward based on the agent's current conditions.
        """
        for ship in self.ships:
            if not done and ship.contains_points(self.agent.getPathInDirection()):
                print("agent is colliding with another ship")
                # print agent position
                print('agent position', self.agent.position)
                done = True
                reward = self.rewards['collision']
                info = self._get_info("collision")
                self.min_distance_to_other_ships = 0
                break
            # calculate the minimum distance to other ships
            dist_to_ship = np.linalg.norm(self.agent.position - ship.position)
            if self.min_distance_to_other_ships is None or dist_to_ship < self.min_distance_to_other_ships:
                self.min_distance_to_other_ships = dist_to_ship
            # print('min distance to other ships', self.min_distance_to_other_ships)
        return done, info, reward

    def _render(self, mode="human"):
        """
        Renders the environment in the specified mode. If mode is "human", the rendering will be displayed
        in a pygame window. If mode is "rgb_array" or "single_rgb_array", the rendering will be returned
        as a NumPy array.

        This method assumes that the following attributes exist:
            - self.window: a pygame window or surface used for rendering
            - self.clock: a pygame time clock used for controlling the framerate
            - self.sea_land_grid: a grid representing the sea and land
            - self._target_location: the target location in the environment
            - self.agent: the agent object in the environment
            - self.ships: list of Ship objects in the environment

        Args:
            mode (str): Rendering mode, one of "human", "rgb_array", or "single_rgb_array". Defaults to "human".

        Returns:
            np.ndarray: If mode is "rgb_array" or "single_rgb_array", returns the rendered environment as a NumPy array.
        """
        assert mode is not None

        if self.window is None:
            #  and mode == "human":
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT))
            else:  # mode in {"rgb_array", "single_rgb_array"}
                self.window = pygame.Surface(
                    (WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(
            (WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT))

        self.window.fill((255, 255, 255))
        canvas.blit(self.background, (0, 0))
        # TODO: if you want to draw the sea and land grid:
        surf = pygame.surfarray.make_surface(self.sea_land_grid)
        canvas.blit(surf, (0, 0))

        # transform the target location to pixel coordinates
        target_location_pix = np.array(
            self.convertEnvCoordinatesToPixCoordinates(self._target_location))
        # Now we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            target_location_pix,
            5,
            0)

        # draw the agent
        self.agent.draw(canvas, self.convertEnvCoordinatesToPixCoordinates, self.scale, (0,40, 255))

        # draw ships
        for ship in self.ships:
            ship.draw(canvas, self.convertEnvCoordinatesToPixCoordinates, self.scale)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            # self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _load_sea_land_grid(self):
        """
        Loads the sea and land grid from a file or creates it if it doesn't exist.
        
        This method reads the sea and land grid from a JSON file, converts the coordinates
        to the environment and pixel coordinates, and creates a grid of size (WINDOW_SIZE_WIDTH,
        WINDOW_SIZE_HEIGHT) with 0 for land and 1 for sea. If the grid file already exists,
        it loads the grid from the file instead of recreating it. The created grid is also
        saved to a file for future use.
        
        Returns:
            numpy.ndarray: A 2D array representing the sea and land grid with 0 for land and 1 for sea.
        """
        with open(os.path.join(__location__, 'sea-land.json'), 'r') as f:
            data = json.load(f)

        # convert in env coordinates
        envCoordinates = []
        for i in range(len(data)):
            envCoordinates.append(self.convertLatLotInEnvCoordinates(
                lat=data[i]["lat"], lon=data[i]["lng"]))
        self.seaPolygonPath = mpltPath.Path(envCoordinates)

        # check if sea-land grid file already exists:
        if os.path.isfile(os.path.join(__location__, 'sea_land_grid_file.npy')):
            # load the grid from file:
            return np.load(os.path.join(__location__,'sea_land_grid_file.npy'))
        # convert in pix coordinates
        pixCoordinates = []
        for i in range(len(data)):
            pixCoordinates.append(self.convertEnvCoordinatesToPixCoordinates(
                envCoordinates[i]))
        # save as numpy array of size of window and 0 if land, 1 if sea
        sea_land_grid = np.zeros(
            (WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT))
        # if not, create it:
        # iterate over all coordinates and set the value to 1 if sea, 0 if land
        sea_land_grid = np.array([[1 if self.seaPolygonPath.contains_point((i, j)) else 0
            for j in range(WINDOW_SIZE_HEIGHT)]
            for i in range(WINDOW_SIZE_WIDTH)])
        # save the grid to file
        sea_land_grid_file = os.path.join(__location__, 'sea_land_grid_file.npy')
        np.save(sea_land_grid_file, sea_land_grid)     
        return sea_land_grid

    def _render_neighborhood(self, agent):
        return self.render_neighborhood(agent, self.ships)

    def render_neighborhood(self, agent, ships):
        """
        Render the neighborhood of the agent.
        The agent is centered in the middle and always facing up.
        The land is therefore rotated to match the agent's direction.
        """
        # print('render neighborhood at : ', self.current_datetime)
        # create new surface of size 100x100 which should represent 1kmx1km of the environment
        neighborhood = pygame.Surface((NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE))
        # fill the surface with red to mark land
        neighborhood.fill((255, 0, 0))
        
        # load path of sea-land polygon
        sea = self.seaPolygonPath

        agentCoordinates = []
        for i in range(len(sea.vertices)):
            # convert vertices in agent coordinates
            positionAgentCoor = self.convertEnvCoordinatesToAgentCoordinates(sea.vertices[i], agent)
            # rotate around the agent's position
            positionRotated = rotate_points(np.array((positionAgentCoor[0], positionAgentCoor[1])), -agent.direction, (0, 0))
            # apply scaling and translate to the origin of the neighborhood
            agentCoordinates.append((positionRotated[0] * NEIGHBORHOOD_SCALE + NEIGHBORHOOD_SIZE/2, positionRotated[1] * NEIGHBORHOOD_SCALE +  NEIGHBORHOOD_SIZE/2))

        draw_polygone(neighborhood, (0, 0, 0), (0,0), agentCoordinates)
        
        # draw ships in neighborhood in green
        for ship in ships:
            # print('draw ship with mmsi: ', ship.mmsi, 'ship position: ', ship.position)
            position = self.convertEnvCoordinatesToAgentCoordinates(ship.position, agent)
            # rotate around the agent's position
            positionRotated = rotate_points(np.array((position[0],position[1])), -agent.direction, (0,0))
            # apply scaling and translate to the origin of the neighborhood
            draw_rect(neighborhood, (0, 255, 0), (
                positionRotated[0] * NEIGHBORHOOD_SCALE + NEIGHBORHOOD_SIZE / 2,
                positionRotated[1] * NEIGHBORHOOD_SCALE + NEIGHBORHOOD_SIZE / 2,
                ship.length * NEIGHBORHOOD_SCALE,
                ship.width * NEIGHBORHOOD_SCALE
                ), ship.direction - agent.direction)

        # print('draw agent: ', agent.mmsi, 'agent position: ', agent.position)
        
        # draw agent in center of the neighborhood in blue
        draw_rect(neighborhood, (0, 0, 255), (NEIGHBORHOOD_SIZE/2, NEIGHBORHOOD_SIZE/2, agent.length * NEIGHBORHOOD_SCALE, agent.width * NEIGHBORHOOD_SCALE), 0)
        return pygame.surfarray.array3d(neighborhood)

        # DISPLAY images for debugging purposes
        # pygame.image.save(neighborhood, "vehicle_surface.png")

    def _get_ships(self):
        """
        Load the ship information for the current datetime from the dataframe and create
        a list of Ship instances representing the other ships in the environment.

        This method resets the 'ships' attribute and populates it with Ship instances
        based on the current datetime. Ship instances are created using position, 
        direction, and other relevant data from the dataframe.
        """
        # reset ships
        self.ships = []
        # get all position data at starting time
        for row in self.df_datetime.get_group(self.current_datetime.strftime('%Y-%m-%d %H:%M:%S')+'+00:00').iterrows():
            # add ship to ships array
            self.ships.append(Ship(position=(row[1]['x'], row[1]['y']), direction=row[1]['direction'], dt=self.dt, mmsi=row[1]['mmsi'], df_index=row[0]))


# =============================================================================
#                                                                               # Static methods
    @staticmethod
    def convertToPxCoordinates(lat, lon):
        """
        Convert latitude and longitude to pixel coordinates.

        :param lat: float, latitude
        :param lon: float, longitude
        :return: tuple, pixel coordinates (x, y)
        """
        return LatLonToPxPy(lat, lon, BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT)

    @staticmethod    
    def convertLatLotInEnvCoordinates(lat=54.35, lon=10.5):
        """
        Convert latitude and longitude to environment coordinates.

        :param lat: float, latitude (default: 54.35)
        :param lon: float, longitude (default: 10.5)
        :return: tuple, environment coordinates (x, y)
        """
        pixCoordinates = LatLonToPxPy(lat, lon, BOUND_LEFT, BOUND_RIGHT,
                                      BOUND_TOP, BOUND_BOTTOM, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT)
        scale = get_scale(BOUND_LEFT, BOUND_RIGHT, BOUND_TOP,
                          BOUND_BOTTOM, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT)
        return PxPyToMeters(pixCoordinates[0], pixCoordinates[1], scale[0], scale[1])
    @staticmethod
    def convertEnvCoordinatesToPixCoordinates(position):
        """
        Convert environment coordinates to pixel coordinates.

        :param position: tuple, environment coordinates (x, y)
        :return: tuple, pixel coordinates (x, y)
        """
        scale = get_scale(BOUND_LEFT, BOUND_RIGHT, BOUND_TOP,
                          BOUND_BOTTOM, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT)
        return MetersToPxPy(position[0], position[1], scale[0], scale[1])

    @staticmethod
    def convertPixCoordinatesToEnvCoordinates(position):
        """
        Convert pixel coordinates to environment coordinates.

        :param position: tuple, pixel coordinates (x, y)
        :return: tuple, environment coordinates (x, y)
        """

        scale = get_scale(BOUND_LEFT, BOUND_RIGHT, BOUND_TOP,
                          BOUND_BOTTOM, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT)
        return PxPyToMeters(position[0], position[1], scale[0], scale[1])

    @staticmethod
    def convertEnvCoordinatesToAgentCoordinates(position, agent):
        """
        Convert environment coordinates to agent-relative coordinates.

        :param position: tuple, environment coordinates (x, y)
        :param agent: object, agent instance
        :return: tuple, agent-relative coordinates (x, y)
        """
        return (position[0] - agent.position[0], position[1] - agent.position[1])

    @staticmethod
    def scale(value, direction = 0):
        """
        Scales the given value based on the window size and coordinate bounds.
        
        Parameters:
            value (float): The value to be scaled.
            direction (int, optional): The direction for scaling, either 0 (x-axis) or 1 (y-axis). Defaults to 0.
            
        Returns:
            float: The scaled value.
        """
        scale = get_scale(BOUND_LEFT, BOUND_RIGHT, BOUND_TOP,
                        BOUND_BOTTOM, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT)
        return value / scale[direction]


