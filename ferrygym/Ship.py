import math

import numpy as np
import pygame
from pygame.math import Vector2
import matplotlib.path as mpltPath

# local imports
from utils.graphics import draw_rect
from constants import  MAX_VELOCITY, MAX_DIRECTION_CHANGE


class Ship:
    """
    Class representing a ship in a simulation.
    """ 
    def __init__(self, position, direction, dt, mmsi='notSet', df_index='notSet', length=40, width=10, velocity=0):
        """
        Initialize a Ship object with given parameters.

        :param position: A tuple representing the ship's initial position (x, y) in meters.
        :param direction: The ship's initial direction in degrees.
        :param dt: The time step for the simulation.
        :param mmsi: The unique Maritime Mobile Service Identity (MMSI) number for the ship (default: 'notSet').
        :param df_index: Index of the ship in a dataframe (default: 'notSet').
        :param length: The ship's length in meters (default: 40).
        :param width: The ship's width in meters (default: 10).
        :param velocity: The ship's initial velocity in meters/second (default: 0).
        """
        # in environment coordinates (meters)
        self.position = Vector2(position)
        self.velocity = velocity # in m/s
        self.direction = direction  # in degrees
        self.length = length  # in meters
        self.width = width  # in meters
        self.dt = dt
        self.mmsi = mmsi
        self.df_index = df_index

    def step(self, action):
        """
        Update the ship's position, given the current velocity and acceleration.
    
        :param action: A tuple containing acceleration (a) and steering (b) values.
        """
        # Actions: acceleration (a), steering (b)
        a, b = action
        # limit direction change
        b = np.clip(b, -MAX_DIRECTION_CHANGE, MAX_DIRECTION_CHANGE)

        self.velocity += a * self.dt
        self.velocity = max(-MAX_VELOCITY,
                              min(self.velocity, MAX_VELOCITY))
        self.direction =  (self.direction + b * self.dt) % 360
        # State integration
        self.position.x = self.position.x + self.velocity * math.sin(math.radians(self.direction)) * self.dt
        self.position.y = self.position.y - self.velocity * math.cos(math.radians(self.direction)) * self.dt

        # steering


    def draw(self, canvas, convertToPxCoordinates, scale, color=(0, 255, 0)):
        """
        Draw the ship on a given canvas.

        :param canvas: The pygame surface to draw the ship on.
        :param convertToPxCoordinates: A function to convert environment coordinates to pixel coordinates.
        :param scale: A scaling factor for the ship's dimensions.
        :param color: The color of the ship (default: (0, 255, 0)).
        """
        # transform the target location to pixel coordinates
        agent_location_pix = np.array(
            convertToPxCoordinates(self.position))
        # Now we draw the agent
        # 1m is very roughly 1px, we draw a ship with the size of 40m 10m, thats about 12m longer then the Schilksee IMO: 8605507
        ship = pygame.Rect(
            (agent_location_pix[0],
            agent_location_pix[1]),
            (scale(self.length, 0), 
            scale(self.width, 1))
        )
        draw_rect(canvas, color, ship, self.direction)

    def getPathInDirection(self):
        """
        Compute the coordinates of the ship's corners in the direction it is facing.

        This method takes into account the ship's position, length, width, and direction. The
        ship is represented as a rectangle, and the method rotates the rectangle according to
        the ship's direction, which is measured in degrees from the north.

        Returns:
            numpy.ndarray: A 4x2 array of the ship's corner coordinates, where each row
            represents a corner, and each column represents the x and y coordinates.
        """

        x, y = self.position
        l = self.length
        w = self.width
        xy = np.array(((x-l/2, y - w/2), (x+l/2, y - w/2),
                      (x + l/2, y + w/2), (x - l/2, y + w/2)))
        # the rectangle is drawn at a 90 degree angle, so we need to rotate it so it is facing north first
        direction_north = self.direction - 90
        s = math.sin(math.radians(direction_north))
        c = math.cos(math.radians(direction_north))
        rot = np.array(((c, -s), (s, c)))
        return (rot @ (xy - (x, y)).T).T + (x, y)

    def contains_points(self, points):
        """
        Check if the ship's path contains any of the given points.

        This method determines if the ship's rectangular path, as computed by the
        getPathInDirection method, contains any of the points provided as input.

        :param points (numpy.ndarray): A numpy array of points, where each row represents a point, and each column represents the x and y coordinates.

        Returns:
            bool: True if any of the points are within the ship's path; False otherwise.
        """
        shipPath = mpltPath.Path(self.getPathInDirection(), None, 2)
        collision = shipPath.contains_points(points)
        if collision.any():
            # print 
            print('collision with ship', self.mmsi, self.getPathInDirection(), points)
            
        return collision.any()
