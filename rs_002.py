"""
This file will be used to help determine depth of
"""
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

# open pipeline
pipe = rs.pipeline()

# configure pipeline

config = rs.config()
WIDTH = 640
HEIGHT = 480
FRAME_RATE = 30
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FRAME_RATE)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FRAME_RATE)

# apply config and start
pipe.start(config)

# wait so many frames for camera to adjust
for x in range(5):
    pipe.wait_for_frames()

# store next frameset for later processing
frameset = pipe.wait_for_frames()
img_color = frameset.get_color_frame()
img_depth = frameset.get_depth_frame()

pipe.stop()

colorizer = rs.colorizer()
colorized_depth = np.asanyarray(img_depth.get_data())
plt.imshow(colorized_depth)
plt.show()