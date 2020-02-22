"""
AGVT COMPUTER VISION FUNCTIONS
"""
import pyrealsense2 as rs
import numpy as np

class Realsense435:
    def __init__(self):
        self.pipe = rs.pipeline()
        self.config = rs.config()

        self.w = 640
        self.h = 480
        self.fr = 30

        self.buf = 10

        self.color = None
        self.depth = None

    def configure_rs(self):
        # save Depth data as z16 (depth data type)
        self.config.enable_stream(rs.stream.depth,
                                  self.w,
                                  self.h,
                                  rs.format.z16,
                                  self.fr)
        # Save Color data as RGB
        self.config.enable_stream(rs.stream.color,
                                  self.w,
                                  self.h,
                                  rs.format.rgb8,
                                  self.fr)

    def open_pipeline(self):
        self.pipe.start(self.config)

    def buffer(self, buf):
        for x in range(buf):
            self.pipe.wait_for_frames()

    def stop_pipe(self):
        self.pipe.stop()

    def get_frames(self):
        """
        Will allign, update and retrieve np frames
        :return:
        RBG Numpy arrays with frames
        Z16 Depth frames
        """
        frame_set = self.pipe.wait_for_frames()
        # create alignment primitive
        align = rs.align(rs.stream.color)
        frame_set = align.process(frame_set)
        # get frames
        color = frame_set.get_color_frame()
        depth = frame_set.get_depth_frame()
        # convert to np arrays
        color = np.asarray(color.get_data())
        depth = np.asarray(depth.get_data())

        return color, depth

    def start_rs(self):
        self.configure_rs()
        self.open_pipeline()
        self.buffer(self.buf)


