import math
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2
import numpy as np

import rs_help

myRealSense = rs_help.MyRealsense()
myRealSense.rs_configure()
myRealSense.rs_start_pipe()

# height and rotation in mm and in radians
height = 130
angle = 0


def get_filtered_color(color_frame, base_color, thresh):
    """
    will filter the class color frame with parameters
    :param color_frame:
    :param base_color:
    :param thresh:
    :return:
    """
    bound_lower = [base_color[0] - thresh[0], base_color[1] - thresh[1], base_color[2] - thresh[2]]
    bound_upper = [base_color[0] + thresh[0], base_color[1] + thresh[1], base_color[2] + thresh[2]]
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
    # return filtered frame
    return cv2.inRange(color_frame, np.array(bound_lower), np.array(bound_upper))

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        myRealSense.rs_align_and_update_frames()
        color_frame = myRealSense.rs_color_frame
        hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2HSV)

        # Mask the color image to get the ground
        groundMask, objectMask = myRealSense.rs_get_ground_obstacle_mask(height, angle)
        groundIMG = cv2.GaussianBlur(cv2.bitwise_and(hsv_frame, hsv_frame, mask=groundMask), (9, 9), 2)
        objectIMG = cv2.bitwise_and(color_frame, color_frame, mask=objectMask)
        groundIMGValue = groundIMG[:, :, 2]
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(groundIMGValue)
        # Stack both images horizontally
        brightestColor = groundIMG[maxLoc[1], maxLoc[0]]
        print(maxLoc)
        print(brightestColor)
        LinesMask = get_filtered_color(groundIMG, brightestColor, [50, 65, 30])
        LinesIMG = cv2.bitwise_and(color_frame, color_frame, mask=LinesMask)
        images = np.hstack((groundIMG, LinesIMG))


        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Normal Image', color_image)
        # cv2.imshow('Depth Image', depth_image)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            plt.imshow(groundIMG)
            plt.show()
            while cv2.waitKey(0) & 0xFF != ord('p'):
                x = 0

finally:

    # Stop streaming
    myRealSense.rs_stop_pipe()