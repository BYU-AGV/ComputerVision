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
height = 122
rotation = (90-0) * math.pi / 180
fov = 10 * math.pi / 180

transVec = np.array([0, 0, 200])
rotMat = np.array([[1, 0, 0],
                  [0, math.cos(-rotation), -math.sin(-rotation)],
                  [0, math.sin(-rotation), math.cos(-rotation)]])

aspectRatio = myGroundDetector.rs_width/myGroundDetector.rs_height

groundFilterIMG = np.zeros((myGroundDetector.rs_height, myGroundDetector.rs_width))
for u in range(myGroundDetector.rs_width):
    for v in range(myGroundDetector.rs_height):
        rayDirection = np.array([(2*(u/myGroundDetector.rs_width)-1)*aspectRatio*math.tan(fov/2),
                                (1-2*(v/myGroundDetector.rs_height))*math.tan(fov/2),
                                -1])
        rayDirection = np.matmul(rayDirection, rotMat)
        magnitude = math.sqrt(rayDirection[0]**2 + rayDirection[1]**2 + rayDirection[2]**2)
        rayDirection = rayDirection/magnitude
        depth = 2**16
        if rayDirection[2] < (-height / 2**16):
            depth = -height / rayDirection[2]

        groundFilterIMG[v][u] = depth
graphicalGFI = cv2.applyColorMap(cv2.convertScaleAbs(groundFilterIMG, alpha=0.03), cv2.COLORMAP_JET)
cv2.namedWindow("Linear Gradient", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Linear Gradient", graphicalGFI)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        myGroundDetector.rs_align_and_update_frames()
        depth_frame = myGroundDetector.rs_depth_frame
        color_frame = myGroundDetector.rs_color_frame

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

        # Subtract the two images
        heightMap = cv2.subtract(depth_frame.astype(np.int32), groundFilterIMG.astype(np.int32))
        heightMapMask = cv2.inRange(heightMap, 0, 2**16)
        filteredIMG = cv2.bitwise_and(color_frame, color_frame, mask=heightMapMask)

        # Stack both images horizontally
        images = np.hstack((filteredIMG, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Normal Image', color_image)
        # cv2.imshow('Depth Image', depth_image)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    myGroundDetector.rs_stop_pipe()