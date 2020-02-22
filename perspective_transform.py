import math
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2
import numpy as np
import rs_help
import time


def get_map(depth):
    MAX_DIM_CM = 20
    ANGLE_RAD = float(50 * np.pi / 180)

    # grab shape
    height, width = depth.shape

    if 'tanLUT' not in get_map.__dict__:
        tanLUT = np.zeros(width)
        for column in range(width):
            tanLUT[column] = np.tan(ANGLE_RAD/2 - (ANGLE_RAD * column/width))

    # create map accumulator for data
    myMap = np.zeros((MAX_DIM_CM, MAX_DIM_CM))

    # update shapes
    height, width = depth.shape

    for row in range(height):
        for column in range(width):
            z = depth[row, column]/10  # grab depth and convert to cm
            if z == 0:  # ignore if depth mask is zero
                continue
            x = tanLUT[column]*z  # get x distance from middle horizontal
            if int(z) >= MAX_DIM_CM or int(z) < 0 or int(abs(x)) >= int(MAX_DIM_CM/2):
                continue  # if depth will fall outside of accumulator, just skip placing
            else:
                myMap[int(MAX_DIM_CM-z), int(MAX_DIM_CM/2 - x)] += 1

    return myMap

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


# number of frames to average together
N = 10


def average_colors(newest_color):
    if 'colorRGB' not in average_colors.__dict__:
        average_colors.colorRGB = np.array([newest_color])

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


myRealSense = rs_help.MyRealsense()
myRealSense.rs_configure()
myRealSense.rs_start_pipe()

# height and rotation in mm and in radians
height = 130
angle = 0
# max depth analyzed
max_depth = 3000
max_depth_y_coord = 0
myRealSense.rs_align_and_update_frames()
groundMask, objectMask = myRealSense.rs_get_ground_obstacle_mask(height, angle)
min_depth = myRealSense.ground_depth_estimation_img[myRealSense.rs_height-1][0]
for v in range(myRealSense.rs_height):
    if myRealSense.ground_depth_estimation_img[myRealSense.rs_height-v-1][0] < max_depth:
        max_depth_y_coord = myRealSense.rs_height-v-1

new_img_width = 2*max_depth*math.tan(58/2 * (math.pi / 180))
new_img_min_width = 2*min_depth*math.tan(58/2 * (math.pi / 180))
new_img_center = new_img_width / 2

# define matrix for perspective transform
original_points = np.float32([[0, max_depth_y_coord], [myRealSense.rs_width, max_depth_y_coord],
                              [0, myRealSense.rs_height], [myRealSense.rs_width, myRealSense.rs_height]])
new_points = np.float32([[0, 0], [new_img_width, 0],
                         [new_img_center - new_img_min_width/2, max_depth], [new_img_center + new_img_min_width/2, max_depth]])
perspective_transform_mat = cv2.getPerspectiveTransform(original_points, new_points)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        myRealSense.rs_align_and_update_frames()

        color_frame = myRealSense.rs_color_frame
        hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2HSV)
        depth_frame = myRealSense.rs_get_depth()
        colored_depth_frame = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

        # Mask the color image to get the ground
        groundMask, objectMask = myRealSense.rs_get_ground_obstacle_mask(height, angle)
        sobely = -cv2.Sobel(objectMask, cv2.CV_64F, 0, 1, ksize=5)
        groundThinMask = cv2.inRange(sobely, 1, 2 ** 16)
        groundThinMask = cv2.erode(groundThinMask, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1)))
        groundThinMask = cv2.dilate(groundThinMask, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1)))
        groundIMG = cv2.GaussianBlur(cv2.bitwise_and(hsv_frame, hsv_frame, mask=groundMask), (5, 5), 2)
        objectIMG = cv2.bitwise_and(color_frame, color_frame, mask=objectMask)
        groundIMGValue = groundIMG[:, :, 2]
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(groundIMGValue)

        # Determine the brightest color used to mask the lines
        brightestColor = groundIMG[maxLoc[1], maxLoc[0]]
        brightestColor = average_colors(brightestColor)
        print(maxLoc)
        print(brightestColor)
        LinesMask = get_filtered_color(groundIMG, brightestColor, [50, 65, 100])
        LinesIMG = cv2.bitwise_and(color_frame, color_frame, mask=LinesMask)

        LinesDepthIMG = cv2.bitwise_and(colored_depth_frame, colored_depth_frame, mask=LinesMask)

        linesOrObject = cv2.bitwise_or(LinesMask, groundThinMask)

        # Perform perspective transform on the lines mask
        transformed_color = cv2.warpPerspective(LinesDepthIMG, perspective_transform_mat, (int(new_img_width), int(max_depth)))
        transformed_lines = cv2.warpPerspective(LinesMask, perspective_transform_mat, (int(new_img_width), int(max_depth)))
        transformed_lines_or_object = cv2.warpPerspective(linesOrObject, perspective_transform_mat, (int(new_img_width), int(max_depth)))
        #plt.subplot(121), plt.imshow(LinesMask), plt.title('Input')
        #plt.subplot(122), plt.imshow(transformed_color), plt.title('Output')
        #plt.show()

        lines_or_object_depth = cv2.bitwise_and(depth_frame, depth_frame, mask=linesOrObject)
        #other_transformed_lines_or_object = get_map(lines_or_object_depth)

        # Stack both images horizontally
        images = np.hstack((linesOrObject, cv2.resize(transformed_lines_or_object, (int(new_img_width * myRealSense.rs_height / max_depth), myRealSense.rs_height))))

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
