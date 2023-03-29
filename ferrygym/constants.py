# CONSTANTS
G = 9.81  # gravity in m/s^2
MAX_ACCElERATION = 0.5 * G
MIN_ACCELERATION = -5 * G
MAX_VELOCITY = 10.0  # in m/s
MAX_DIRECTION = 180.0  # in degrees
MAX_DIRECTION_CHANGE = 10.0  # in degrees

# Control the size of the neighborhood of the agent
NEIGHBORHOOD_SIZE = 100 
NEIGHBORHOOD_SCALE = 0.1 # to match 1m in environment coordinates