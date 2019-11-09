"""
This file is to help do lane detection
"""

import cv_help
import rs_help
import cv2 as cv
import numpy as np
import logging
from matplotlib import pyplot as plt
import pyrealsense2 as rs

def region_of_interest(edges):
    # get shape dimensions
    height, width = edges.shape
    # get array of zeros (blank image) with same dimensions as what is passed in
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv.fillPoly(mask, polygon, 255)
    cropped_edges = cv.bitwise_and(edges, mask)
    return cropped_edges


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                    np.array([]), minLineLength=75, maxLineGap=10)

    return line_segments

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


if __name__ == '__main__':
    # load image
    image = cv.imread('lines.png')

    # get gray image
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # get hsv to find values for color
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    line_color = [40, 30, 255]
    line_thresh = [30, 50, 50]

    # filter lines
    line_mask = cv_help.cv_filter_hsv(image_hsv, line_color, line_thresh)

    # # erode and diilate mask
    line_mask = cv_help.cv_erode(line_mask, 2)
    line_mask = cv_help.cv_dilate(line_mask, 10)
    line_mask = cv_help.cv_erode(line_mask, 8)

    # mask with original
    masked_image_gray = cv.bitwise_and(image_gray, line_mask)

    # add a gaussian blur to help with canny edge detection
    kernel_size = 9  # must be an odd number
    gauss_gray = cv.GaussianBlur(masked_image_gray, (kernel_size, kernel_size), 0)

    # canny edge detection
    low_thresh = 50
    high_thresh = 150
    canny_edges = cv.Canny(gauss_gray, low_thresh, high_thresh)

    # create ROI (region of interest for the lines)
    # in self-driving cars, its usually a polygon directly infront of the car
    # for us it will be the lower half of the screen
    cropped_edges = region_of_interest(canny_edges)

    # find line segments
    line_segments = detect_line_segments(cropped_edges)

    lanes = average_slope_intercept(image, line_segments)
    print(lanes)

    # for x in range(len(line_segments)):
    #     line = line_segments[x][0]
    #     print('line number: %d' % x)
    #     print(line)
    #     print(cv_help.cv_get_line_length_pixel(line))
    #     print('----------------------')


    # # display image
    # cv.imshow('image', image)
    # cv.imshow('line_mask', cropped_edges)
    # cv.waitKey(0)

    # cv.imshow('current image', cropped_edges)
    # cv.waitKey(0)

    # # display matplot
    # plt.imshow(image_hsv)
    # plt.show()