"""
This file contains helper functions to make interfacing with the Intel Realsense D435 easier for our purposes
"""


from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import math

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
    rs_get_ground_obstacle_mask()
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

        # variables for storing previously used ground plane estimation image so it doesn't have to recompute it each
        # frame
        self.height_above_ground = 0
        self.angle_below_horizontal = 0
        self.ground_depth_estimation_img = np.zeros((self.rs_height, self.rs_width))

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
        return cv.inRange(cv.cvtColor(self.rs_color_frame, cv.COLOR_RGB2HSV),
                          np.array(bound_lower), np.array(bound_upper))

    @staticmethod
    def rs_get_filtered_color_of_frame(color_frame, base_color, thresh):
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
        return cv.inRange(color_frame, np.array(bound_lower), np.array(bound_upper))

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

    # generates a mask that only allows the stuff at the same depth as the ground pass
    # also generates a mask that allows anything above the ground and below the sky pass
    # the first call will be slow because it has to generate a new image estimating the depth of the ground
    # height is the height of the camera above the ground in mm
    # angle is the angle below the horizontal of the camera in degrees
    # only works if rs_align_and_update_frames() is called first
    def rs_get_ground_obstacle_mask(self, height, angle):
        if height != self.height_above_ground or angle != self.angle_below_horizontal:
            # set variables so it won't recalculate next time
            self.height_above_ground = height
            self.angle_below_horizontal = angle

            # recalculate the ground depth estimation image

            # initial variables set up
            aspect_ratio = self.rs_width / self.rs_height
            fov = 85 * math.pi / 180
            rotation = (90 - angle) * math.pi / 180
            rot_mat = np.array([[1, 0, 0],
                               [0, math.cos(-rotation), -math.sin(-rotation)],
                               [0, math.sin(-rotation), math.cos(-rotation)]])

            # get depth for particular row by estimating depth of center pixel
            for v in range(self.rs_height):
                ray_direction = np.array([-1 * aspect_ratio * math.tan(fov / 2),
                                         (1 - 2 * (v / self.rs_height)) * math.tan(fov / 2),
                                         -1])
                ray_direction = np.matmul(ray_direction, rot_mat)
                magnitude = math.sqrt(ray_direction[0] ** 2 + ray_direction[1] ** 2 + ray_direction[2] ** 2)
                ray_direction = ray_direction / magnitude
                estimated_depth = 2 ** 16
                if ray_direction[2] < (-height / 2 ** 16):
                    estimated_depth = -height / ray_direction[2]

                # set all of the pixels in the row to the estimated depth
                for u in range(self.rs_width):
                    self.ground_depth_estimation_img[v][u] = estimated_depth

        # now the estimation image is updated, so we compare it with the last depth image taken to generate the mask

        height_map = cv.subtract(self.rs_depth_frame.astype(np.int32),
                                 self.ground_depth_estimation_img.astype(np.int32))
        ground_mask = cv.inRange(height_map, 1, 2 ** 16)
        object_mask = cv.inRange(height_map, -2**16 + 100, -1)

        return ground_mask, object_mask

# Runs super slowly. Don't use
    def rs_get_line_points(self, height, angle):
        def average_colors(newest_color):
            if 'colorRGB' not in average_colors.__dict__:
                average_colors.colorRGB = np.array([newest_color])

            N = 200

            # create / update colorRGB array
            array_size = average_colors.colorRGB.shape[0]
            if array_size <= N:
                average_colors.colorRGB = np.vstack((average_colors.colorRGB, newest_color))
            else:
                for i in range(0, N - 1):
                    average_colors.colorRGB[i] = average_colors.colorRGB[i + 1]
                average_colors.colorRGB[N - 1] = newest_color

            # average everything together
            row, col = average_colors.colorRGB.shape

            b = np.convolve(average_colors.colorRGB[:, 0], np.ones(row) / row, mode='valid')
            g = np.convolve(average_colors.colorRGB[:, 1], np.ones(row) / row, mode='valid')
            r = np.convolve(average_colors.colorRGB[:, 2], np.ones(row) / row, mode='valid')
            averaged_color = np.concatenate((b, g, r), axis=0)
            return averaged_color

        line_points = []

        color_frame = self.rs_color_frame
        hsv_frame = cv.cvtColor(color_frame, cv.COLOR_RGB2HSV)
        depth_frame = self.rs_get_depth()
        colored_depht_frame = cv.applyColorMap(cv.convertScaleAbs(depth_frame, alpha=0.03), cv.COLORMAP_JET)

        # Mask the color image to get the ground
        groundMask, objectMask = self.rs_get_ground_obstacle_mask(height, angle)
        groundIMG = cv.GaussianBlur(cv.bitwise_and(hsv_frame, hsv_frame, mask=groundMask), (5, 5), 2)
        groundIMGValue = groundIMG[:, :, 2]
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(groundIMGValue)
        # Stack both images horizontally
        brightestColor = groundIMG[maxLoc[1], maxLoc[0]]
        brightestColor = average_colors(brightestColor)
        print(maxLoc)
        print(brightestColor)
        LinesMask = self.rs_get_filtered_color_of_frame(groundIMG, brightestColor, [50, 65, 100])
        LinesIMG = cv.bitwise_and(color_frame, color_frame, mask=LinesMask)

        for x in range(0, self.rs_width):
            for y in range(0, self.rs_height):
                if LinesMask[y, x] == 255:
                    line_points.append(rs.rs2_deproject_pixel_to_point(
                        self.rs_pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics(),
                                                                       [y, x], float(depth_frame[y, x])))

        return line_points


if __name__ == '__main__':
    print('hello world')
