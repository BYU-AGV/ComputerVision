"""
This file is supposed to be added support to process photos with computer vision
"""

import cv2 as cv
import numpy as np
import math

def cv_filter_hsv(src, color, thresh):
    """
    Will return a filtered image from the given parameters
    :param src:
    :param color:
    :param thresh:
    :return:
    """
    bound_lower = [color[0]-thresh[0], color[1]-thresh[1], color[2]-thresh[2]]
    bound_upper = [color[0]+thresh[0], color[1]+thresh[1], color[2]+thresh[2]]

    if bound_lower[0] < 0:
        bound_lower[0] = 0
    if bound_lower[1] < 0:
        bound_lower[1] = 0
    if bound_lower[2] < 0:
        bound_lower[2] = 0
    if bound_upper[0] > 179:
        bound_upper[0] = 179
    if bound_upper[1] > 255:
        bound_upper[1] = 255
    if bound_upper[2] > 255:
        bound_upper[2] = 255

    return cv.inRange(src, np.array(bound_lower), np.array(bound_upper))

def cv_erode(src, iterations):
    return cv.erode(src, None, iterations=iterations)

def cv_dilate(src, iterations):
    return cv.dilate(src, None, iterations=iterations)

def cv_get_line_length_pixel(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    a = x2 - x1
    b = y2 - y1
    return abs(math.sqrt((a**2)+(b**2)))
