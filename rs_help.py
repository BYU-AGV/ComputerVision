"""
This file contains helper functions to make interfacing with the Intel Realsense D435 easier for our purposes
"""


from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import pyrealsense2 as rs

class MyRealsense:
    """
    Creating an object in the class will create the necessary objects needed to open communication with the intel
    realsense

    ############################
    Functions:
    rs_configure()
    rs_start_pipe()
    rs_lens_buffer(num_frames)
    rs_stop_pipe()
    rs_align_and_update_frames()
    ############################

    """
    def __init__(self):
        # pipeline class variable
        self.rs_pipeline = rs.pipeline()
        # flag to indicate that pipeline is open
        self.rs_pipeline_is_open = False
        # configuration class variable
        self.rs_config = rs.config()
        # class variables for frames
        self.rs_color_frame = None
        self.rs_depth_frame = None

        # default configurations
        self.rs_width = 640
        self.rs_height = 480
        self.rs_frame_rate = 30

    def rs_configure(self):
        try:
            self.rs_config.enable_stream(rs.stream.depth,
                                         self.rs_width,
                                         self.rs_height,
                                         rs.format.z16,
                                         self.rs_frame_rate)
            self.rs_config.enable_stream(rs.stream.color,
                                         self.rs_width,
                                         self.rs_height,
                                         rs.format.bgr8,
                                         self.rs_frame_rate)
            return True
        except:
            return False

    def rs_start_pipe(self):
        # check flag
        if self.rs_pipeline_is_open:
            return False
        else:
            try:
                self.rs_pipeline.start(self.rs_config)
                # update flag
                self.rs_pipeline_is_open = True
                return True
            except:
                return False

    def rs_lens_buffer(self, num_frames):
        if not self.rs_pipeline_is_open:
            return False
        try:
            for x in range(num_frames):
                self.rs_pipeline.wait_for_frames()
            return True
        except:
            return False

    def rs_stop_pipe(self):
        if not self.rs_pipeline_is_open:
            return False
        else:
            try:
                self.rs_pipeline.stop()
                # update flag
                self.rs_pipeline_is_open = False
                return True
            except:
                return False

    def rs_align_and_update_frames(self):
        # wait for frames
        try:
            # get frameset
            frame_set = self.rs_pipeline.wait_for_frames()
            # create alignment primitive with color as its target stream
            align = rs.align(rs.stream.color)
            frame_set = align.process(frame_set)
            # update color and depth frames
            color_frame = frame_set.get_color_frame()
            depth_frame = frame_set.get_depth_frame()
            # convert to numpy frames
            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())
            # update class variables
            self.rs_color_frame = color_np
            self.rs_depth_frame = depth_np

            return True

        except:
            return False

    def rs_get_color(self):
        return self.rs_color_frame

    def rs_get_depth(self):
        return self.rs_depth_frame

    def rs_get_filtered_color(self, base_color, thresh):
        """
        will filter the class color frame with parameters
        :param base_color:
        :param thresh:
        :return:
        """
        bound_lower = [base_color[0]-thresh[0], base_color[1]-thresh[1], base_color[2]-thresh[2]]
        bound_upper = [base_color[0]+thresh[0], base_color[1]+thresh[1], base_color[2]+thresh[2]]
        if bound_lower[0] < 0:
            bound_lower[0] = 0
        if bound_lower[1] < 0:
            bound_lower[0] = 0
        if bound_lower[2] < 0:
            bound_lower[0] = 0
        if bound_upper[0] > 179:
            bound_upper[0] = 179
        if bound_upper[1] > 255:
            bound_upper[1] = 255
        if bound_upper[2] > 255:
            bound_upper[2] = 179
        # return filtered frame
        return cv.inRange(self.rs_color_frame, np.array(bound_lower), np.array(bound_upper))

    def rs_get_filtered_depth(self, depth, thresh):
        """
        will filter the class depth frame with parameters
        :param depth:
        :param thresh:
        :return:
        """
        bound_lower = [depth[0]-thresh[0]]
        bound_upper = [depth[0]+thresh[0]]
        # check lower
        if bound_lower[0] < 0.0:
            bound_lower[0] = 0.0
        # return filtered frame
        return cv.inRange(self.rs_depth_frame, np.array(bound_lower), np.array(bound_upper))






if __name__ == '__main__':
    print('hello world')



