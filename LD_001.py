"""
this file will be used to read map out lanes
"""
# Libraries
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import helper_rs as h
import sys

# create realsense object
try:
    myRS = h.MyRealsense()
except:
    sys.exit('COULD NOT CREATE REALSENSE OBJECT')
# configure object
# not needed because these values are the same as the defaults
# this is just to show how to do it
# these parameters must be changed before the realsense is configured
try:
    myRS.width = 640
    myRS.height = 480
    myRS.frame_rate = 30
except:
    sys.exit('COULD NOT UPDATE CLASS CONFIGURATIONS')

# configure
if not myRS.MyRealsense_configure():
    sys.exit('COULD NOT CONFIGURE REALSENSE')

# start pipeling
if not myRS.MyRealsense_start_pipe():
    sys.exit('COULD NOT START PIPELINE')

# wait for shutter lense
if not myRS.MyRealsense_wait_for_lense(5):
    myRS.MyRealsense_stop_pipe()
    sys.exit('COULD NOT WAIT FOR LENSE TO CONFIGURE')

distance_base = [1.00*10**3]
distance_thresh = [0.05*10**3]

try:
    while 1:
        color_frame = myRS.MyRealsense_get_color_frame()
        depth_frame = myRS.MyRealsense_get_depth_frame()

        # filter depth frame
        filtered_depth = myRS.MyRealsense_filter_depth(depth_frame, distance_base, distance_thresh)

        # erode and dilate frames a bit
        # noise reduction
        filtered_depth = myRS.MyRealsense_erode_frame(filtered_depth, 1)
        # dilate
        filtered_depth = myRS.MyRealsense_dilate_frame(filtered_depth, 4)

        masked_frame = cv.bitwise_and(color_frame, color_frame, mask=filtered_depth)


        combined_frame = np.hstack((color_frame, masked_frame))

        cv.namedWindow('RealSense', cv.WINDOW_AUTOSIZE)
        cv.imshow('RealSense', combined_frame)
        cv.waitKey(1)
finally:
    myRS.MyRealsense_stop_pipe()





