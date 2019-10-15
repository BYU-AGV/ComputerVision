"""
Name: practice001.py
Date: 10/12/2019
Created By: wardk14

    To practice colors and image processing in OpenCV

"""

# import opencv as cv
import cv2 as cv
# import numpy
import numpy as np
# imnport matplot lib
from matplotlib import pyplot as plt

# opens and saves image as local variable 'img'
img = cv.imread('PracticeTrack.jpg')

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_bound = np.array([0, 0, 200])
upper_bound = np.array([179, 120, 255])

mask = cv.inRange(img_hsv, lower_bound, upper_bound)
# mask_inv = cv.bitwise_not(mask)


# img = cv.bitwise_and(img, img, mask=mask)

# ERODING and DILATING
mask_E1 = cv.erode(mask, None, iterations=1)

mask_E1 = cv.dilate(mask_E1, None, iterations=2)

mask_better = cv.erode(mask_E1, None, iterations=1)

# Look for orange
lower_bound_orange = np.array([0, 200, 150])
upper_bound_orange = np.array([20, 255, 255])

mask_orange = cv.inRange(img_hsv, lower_bound_orange, upper_bound_orange)
mask_orange = cv.erode(mask_orange, None, iterations=2)
mask_orange = cv.dilate(mask_orange, None, iterations=2)


orange_and_image = cv.bitwise_and(img, img, mask=mask_orange)

total_frame = cv.bitwise_and(mask_better, mask_better, mask=mask_better)
# total_frame = cv.bitwise_or(mask_better, mask_better, mask=mask_orange)




# displaying image in window
# cv.imshow('image', img)
# cv.imshow('imgage rgb', img_rgb)
plt.imshow(img_hsv)
plt.show()
# cv.imshow('mask', mask)
# cv.imshow('mask_better', mask_better)
# cv.imshow('mask orange', orange_and_image)
# cv.imshow('final frame', total_frame)




# wait for key press
cv.waitKey(0)
# close all windows
cv.destroyAllWindows()