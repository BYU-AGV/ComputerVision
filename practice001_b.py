"""
Name: practice001_b.pu
Date: 10/14/2019
Created By: wardk14

    This file is a clean version of the homework that was assigned in the 10/12/2019 meeting
"""

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


"""
########################################################################################################################
###   DEFINITIONS   ###   DEFINITIONS   ###   DEFINITIONS   ###   DEFINITIONS   ###   DEFINITIONS   ###   DEFINITIONS   
########################################################################################################################
"""


HUE_THRESH = 15
SAT_THRESH = 75
VAL_THRESH = 75

W_HUE_THRESH = 15
W_SAT_THRESH = 65
W_VAL_THRESH = 30

LOWER = 0
UPPER = 1

ORANGE = [15, 200, 200]
BLUE = [110, 235, 150]
RED = [80, 250, 240]
WHITE = [35, 65, 230]

LINE_COLOR = (0, 0, 255)

"""
########################################################################################################################
###   FUNCTIONS   ###   FUNCTIONS   ###   FUNCTIONS   ###   FUNCTIONS   ###   FUNCTIONS   ###   FUNCTIONS   ###   FUNCTI
########################################################################################################################
"""


def get_bound(color, bound):
    hue = color[0]
    sat = color[1]
    val = color[2]

    # get lower bound
    if bound == LOWER:
        if hue - HUE_THRESH < 0:
            hue = 0
        else:
            hue -= HUE_THRESH
        if sat - SAT_THRESH < 0:
            sat = 0
        else:
            sat -= SAT_THRESH
        if val - VAL_THRESH < 0:
            val = 0
        else:
            val -= VAL_THRESH
    # get upper bound
    elif bound == UPPER:
        if hue + HUE_THRESH > 179:
            hue = 179
        else:
            hue += HUE_THRESH
        if sat + SAT_THRESH > 255:
            sat = 255
        else:
            sat += SAT_THRESH
        if val + VAL_THRESH > 255:
            val = 255
        else:
            val += VAL_THRESH
    else:
        hue = 0
        sat = 0
        val = 0
    # return array
    # print("Org Color  Hue: %d, Sat: %d, Val: %d" % (color[0], color[1], color[2]))
    # print("Bound: %d, Hue: %d, Sat: %d, Val: %d\n" % (bound, hue, sat, val))
    return np.array([hue, sat, val])


def get_bound_white(color, bound):
    hue = color[0]
    sat = color[1]
    val = color[2]

    # get lower bound
    if bound == LOWER:
        if hue - W_HUE_THRESH < 0:
            hue = 0
        else:
            hue -= W_HUE_THRESH
        if sat - W_SAT_THRESH < 0:
            sat = 0
        else:
            sat -= W_SAT_THRESH
        if val - W_VAL_THRESH < 0:
            val = 0
        else:
            val -= W_VAL_THRESH
    # get upper bound
    elif bound == UPPER:
        if hue + W_HUE_THRESH > 179:
            hue = 179
        else:
            hue += W_HUE_THRESH
        if sat + SAT_THRESH > 255:
            sat = 255
        else:
            sat += W_SAT_THRESH
        if val + W_VAL_THRESH > 255:
            val = 255
        else:
            val += W_VAL_THRESH
    else:
        hue = 0
        sat = 0
        val = 0
    # print("Org Color  Hue: %d, Sat: %d, Val: %d" % (color[0], color[1], color[2]))
    # print("Bound: %d, Hue: %d, Sat: %d, Val: %d\n" % (bound, hue, sat, val))
    return np.array([hue, sat, val])

"""
########################################################################################################################
###   MAIN   ###   MAIN   ###   MAIN   ###   MAIN   ###   MAIN   ###   MAIN   ###   MAIN   ###   MAIN   ###   MAIN   ###
########################################################################################################################
"""


# save image
img = cv.imread('PracticeTrack.jpg')
# resize image
img = cv.resize(img, None, fx=0.5, fy=0.5)

# convert image to hsv scale
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# ORANGE
# create orange mask
mask_orange = cv.inRange(img_hsv, get_bound(ORANGE, LOWER), get_bound(ORANGE, UPPER))
# errode and dilate mask
mask_orange = cv.erode(mask_orange, None, iterations=4)
mask_orange = cv.dilate(mask_orange, None, iterations=12)
mask_orange = cv.erode(mask_orange, None, iterations=8)


# BLUE
mask_blue = cv.inRange(img_hsv, get_bound(BLUE, LOWER), get_bound(BLUE, UPPER))
# errode and dilate mask
mask_blue = cv.erode(mask_blue, None, iterations=4)
mask_blue = cv.dilate(mask_blue, None, iterations=4)

# RED
# because red is on both sides of the hue spectra, we will invert the colors and search for cyan
img_inv = cv.bitwise_not(img)
img_inv_hsv = cv.cvtColor(img_inv, cv.COLOR_BGR2HSV)

mask_red = cv.inRange(img_inv_hsv, get_bound(RED, LOWER), get_bound(RED, UPPER))
# errode and dilate mask
mask_red = cv.erode(mask_red, None, iterations=4)
mask_red = cv.dilate(mask_red, None, iterations=4)

# WHITE
mask_white = cv.inRange(img_hsv, get_bound_white(WHITE, LOWER), get_bound_white(WHITE, UPPER))

# errode and dilate mask
mask_white = cv.erode(mask_white, None, iterations=1)
mask_white = cv.dilate(mask_white, None, iterations=4)


# MAP OVER ORIGINAL FRAME

# bitwise or all the masks together
mask_compiled = cv.bitwise_or(mask_orange, mask_red)
mask_compiled = cv.bitwise_or(mask_compiled, mask_blue)
mask_compiled = cv.bitwise_or(mask_compiled, mask_white)

# bitwise and mask with original image
masked_img = cv.bitwise_and(img, img, mask=mask_compiled)


# # add image to a pyplot to find bound
# plt.imshow(img_hsv)
# # show plot
# plt.show()
#
# # add image to a pyplot to find bound - INVERTED
# plt.imshow(img_inv_hsv)
# # show plot
# plt.show()

# # # show image through cv package
cv.imshow("Original Image", img)
cv.imshow("Orange Mask", mask_orange)
cv.imshow("Blue Mask", mask_blue)
cv.imshow("Red Mask", mask_red)
cv.imshow("White Mask", mask_white)
cv.imshow("Compiled Masks", mask_compiled)
cv.imshow("Masked Image", masked_img)
cv.waitKey(0)
cv.destroyAllWindows()
