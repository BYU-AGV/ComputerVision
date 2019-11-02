###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

# open pipeline
pipeline = rs.pipeline()
# create configuration object for pipeline to change configurations
config = rs.config()
# configure depth and color streams
WIDTH = 640
HEIGHT = 480
FRAME_RATE = 30
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FRAME_RATE)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FRAME_RATE)

# Start streaming with new configuration
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # if no frame is received, skip everything else
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Normal Image', color_image)
        # cv2.imshow('Depth Image', depth_image)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()