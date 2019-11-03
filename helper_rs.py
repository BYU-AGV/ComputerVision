"""
this file contains a custom class for the intelrealsense
"""

import pyrealsense2 as rs
import numpy as np
import cv2 as cv


class MyRealsense:
    def __init__(self):
        # pipeline class object
        self.pipeline = rs.pipeline()
        # configuration
        self.class_config = rs.config()

        # default configurations
        self.width = 640
        self.height = 480
        self.frame_rate = 30


    def MyRealsense_configure(self):
        try:
        # update configurations
            self.class_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.frame_rate)
            self.class_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.frame_rate)
            return True
        except:
            return False

    def MyRealsense_start_pipe(self):
        try:
            self.pipeline.start(self.class_config)
            return True
        except:
            return False

    def MyRealsense_wait_for_lense(self, num_frames):
        try:
            for x in range(num_frames):
                self.pipeline.wait_for_frames()
            return True
        except:
            return False

    def MyRealsense_stop_pipe(self):
        try:
            self.pipeline.stop()
            return True
        except:
            return False

    def MyRealsense_get_color_frame(self):
        try:
            frameset = self.pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            # convert to np array
            color_image = np.asanyarray(color_frame.get_data())
            return color_image
        except:
            print(' -> COULD NOT GET COLOR FRAME')

    def MyRealsense_get_depth_frame(self):
        try:
            frameset = self.pipeline.wait_for_frames()
            depth_frame = frameset.get_depth_frame()
            # convert to np array
            depth_image = np.asanyarray(depth_frame.get_data())
            return depth_image
        except:
            print(' -> COULD NOT GET DEPTH FRAME')

    def MyRealsense_filter_depth(self, src, val, thresh):
        try:
            # grab boundries
            val_low = val[0] - thresh[0]
            if val_low <= 0:
                val_low = 0
            val_high = val[0] + thresh[0]

            return cv.inRange(src, val_low, val_high)
        except:
            print(' -> COULD NOT FILTER DEPTH')

    def MyRealsense_filter_hsv(self, src, val, thresh):
        try:
            val_low = [val[0]-thresh[0], val[1]-thresh[1], val[2]-thresh[2]]
            if val_low[0] < 0:
                val_low[0] = 0
            if val_low[1] < 0:
                val_low[1] = 0
            if val_low[2] < 0:
                val_low[2] = 0
            val_high = [val[0] + thresh[0], val[1] + thresh[1], val[2] + thresh[2]]
            if val_high[0] > 179:
                val_high[0] = 179
            if val_high[1] > 255:
                val_high[1] = 255
            if val_high[2] > 255:
                val_high[2] = 255
            # create bounds
            bound_low = np.array(val_low)
            bound_high = np.array(val_high)

            return cv.inRange(src, bound_low, bound_high)
        except:
            print(' -> COULD NOT FILTER HSV')


    def MyRealsense_erode_frame(self, src, iterations):
        return cv.erode(src, None, iterations=iterations)


    def MyRealsense_dilate_frame(selfself, src, iterations):
        return cv.dilate(src, None, iterations=iterations)

