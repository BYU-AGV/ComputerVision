import math
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2
import numpy as np

import rs_help

myGroundDetector = rs_help.MyRealsense()
myGroundDetector.rs_configure()
myGroundDetector.rs_start_pipe()

# height and rotation in mm and in radians
height = 130
angle = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        myGroundDetector.rs_align_and_update_frames()
        color_frame = myGroundDetector.rs_color_frame

        # Mask the color image to get the ground
        groundMask, objectMask = myGroundDetector.rs_get_ground_obstacle_mask(height, angle)
        groundIMG = cv2.bitwise_and(color_frame, color_frame, mask=groundMask)
        objectIMG = cv2.bitwise_and(color_frame, color_frame, mask=objectMask)
        LinesMask = myGroundDetector.rs_get_filtered_color([35, 65, 230], [15, 65, 30])
        LinesIMG = cv2.bitwise_and(groundIMG, groundIMG, mask=LinesMask)

        # Stack both images horizontally
        images = np.hstack((groundIMG, objectIMG))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Normal Image', color_image)
        # cv2.imshow('Depth Image', depth_image)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    myGroundDetector.rs_stop_pipe()