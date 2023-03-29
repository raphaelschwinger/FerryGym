import numpy as np
import pygame
import math


def draw_rect(screen, colour, rect, direction=0, thickness=0):
    x, y, l, w = rect
    xy = np.array(((x-l/2, y - w/2), (x+l/2, y - w/2),
                  (x + l/2, y + w/2), (x - l/2, y + w/2)))
    # the rectangle is drawn at a 90 degree angle, so we need to rotate it so it is facing north first
    direction_north = direction - 90
    xy = rotate_points(xy, direction_north, (x, y))
    return pygame.draw.polygon(screen, colour, xy, thickness)


def draw_polygone(screen, colour, destination, points, direction=0, thickness=0, scale=1):
    x,y = destination
    xy = np.array(points)
    # scale the points
    xy = xy * scale
    # rotate the points
    xy = rotate_points(xy, direction, (x,y))
    return pygame.draw.polygon(screen, colour, xy, thickness)

def rotate_points(points, direction, origin=(0,0)):
    x,y = origin
    s = math.sin(math.radians(direction))
    c = math.cos(math.radians(direction))
    rot = np.array(((c, -s), (s, c)))
    return (rot @ (points - (x, y)).T).T + (x, y)