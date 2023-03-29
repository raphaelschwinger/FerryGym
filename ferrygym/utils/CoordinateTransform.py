import math

# Source: https://gis.stackexchange.com/questions/395293/convert-from-latitude-longitude-to-geotiff-screen-pixel-x-y-coordinates-in-pyth


def LatLonToPxPy(latitude, longitude, bound_left, bound_right, bound_top, bound_bottom, width, height):
    """
    Convert latitude and longitude to pixel coordinates on an image.
    
    :param latitude: Latitude of the point.
    :param longitude: Longitude of the point.
    :param bound_left: Left boundary longitude of the image.
    :param bound_right: Right boundary longitude of the image.
    :param bound_top: Top boundary latitude of the image.
    :param bound_bottom: Bottom boundary latitude of the image.
    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :return: Tuple of (x, y) pixel coordinates of the point on the image.
    """
    px, py = longitude, latitude
    px_pc = (px - bound_left) / (bound_right - bound_left)
    py_pc = (bound_top - py) / (bound_top - bound_bottom)
    return (px_pc*width, py_pc*height)

def PxPyToMeters(px, py, scale_x, scale_y):
    """
    Convert pixel coordinates to coordinates in meters from the top left corner of the image.

    :param px: X-coordinate in pixels.
    :param py: Y-coordinate in pixels.
    :param scale_x: Scaling factor for the x-axis.
    :param scale_y: Scaling factor for the y-axis.
    :return: List of [x, y] coordinates in meters from the top left corner of the image.
    """
    return [px*scale_x, py*scale_y]

def MetersToPxPy(x, y, scale_x, scale_y):
    """
    Convert coordinates in meters to pixel coordinates.

    :param x: X-coordinate in meters.
    :param y: Y-coordinate in meters.
    :param scale_x: Scaling factor for the x-axis.
    :param scale_y: Scaling factor for the y-axis.
    :return: Tuple of (x, y) pixel coordinates.
    """
    return (x/scale_x, y/scale_y)

def get_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface in meters.

    :param lat1: Latitude of the first point.
    :param lon1: Longitude of the first point.
    :param lat2: Latitude of the second point.
    :param lon2: Longitude of the second point.
    :return: Distance between the two points in meters.
    """
    R = 6371000  # radius of the earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2-lat1)
    delta_lambda = math.radians(lon2-lon1)

    a = math.sin(delta_phi/2.0) * math.sin(delta_phi/2.0) + math.cos(phi1) * \
        math.cos(phi2) * math.sin(delta_lambda/2.0) * \
        math.sin(delta_lambda/2.0)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def get_scale(bound_left, bound_right, bound_top, bound_bottom, width, height):
    """
    Calculate the scaling factor for the x and y axes.
    
    :param bound_left: Left boundary longitude of the image.
    :param bound_right: Right boundary longitude of the image.
    :param bound_top: Top boundary latitude of the image.
    :param bound_bottom: Bottom boundary latitude of the image.
    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :return: Tuple of (scale_x, scale_y) scaling factors for the x and y axes.
    """
    return get_distance(bound_top, bound_left, bound_top, bound_right) / width,  get_distance(bound_top, bound_left, bound_bottom, bound_left) / height
