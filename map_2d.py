import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MAX_DIM_CM = 100
ANGLE_RAD = float(50*np.pi/180)


def filter_color(hsv, color, thresh):
    """
    Will take an hsv image and filter it for color
    :param hsv:
    :param color:
    :param thresh:
    :return: filtered image
    """
    lower_bound = (color[0]-thresh[0], color[1]-thresh[1], color[2]-thresh[2])
    upper_bound = (color[0]+thresh[0], color[1]+thresh[1], color[2]+thresh[2])

    hsv = cv.inRange(hsv, lower_bound, upper_bound)

    return hsv/255


def get_map(color, depth):
    # grab shape
    height, width, channels = color.shape

    # grab lower half of images
    color = color[int(height/2):, :, :]
    depth = depth[int(height/2):, :]

    # convert color to hsv
    hsv = cv.cvtColor(color, cv.COLOR_RGB2HSV)

    # filter for colors
    # filter for lines

    # filter for cans
    cans = filter_color(hsv=hsv, color=[10, 240, 130], thresh=[5, 30, 50])

    # yellow cones
    y_cones = filter_color(hsv=hsv, color=[25, 140, 160], thresh=[15, 20, 25])


    # green cones
    # g_cones = filter_color(hsv=hsv, color=[55, 255, 21], thresh=[15, 70, 25])


    # plt.imshow(hsv)
    # plt.show()


    # combine masks and reduce to binary
    mask = cans + y_cones
    high_val = (mask != 0)
    mask[high_val] = 1

    # mask obstacle with depth
    depth = depth*mask

    # create map accumulator for data
    myMap = np.zeros((MAX_DIM_CM, MAX_DIM_CM))

    # update shapes
    height, width = depth.shape

    for row in range(height):
        for column in range(width):
            z = depth[row, column]/10  # grab depth and convert to cm
            if z == 0:  # ignore if depth mask is zero
                continue
            theta = ANGLE_RAD * column/width  # calculate pixel angle with respect to origin
            theta = ANGLE_RAD/2 - theta  # shift perspective to that of the center
            x = np.tan(theta)*z  # get x distance from middle horizontal
            if int(z) >= MAX_DIM_CM or int(z) < 0 or int(abs(x)) >= int(MAX_DIM_CM/2):
                continue  # if depth will fall outside of accumulator, just skip placing
            else:
                myMap[int(z), int(MAX_DIM_CM/2 + x)] += 1

    return myMap, mask, color


