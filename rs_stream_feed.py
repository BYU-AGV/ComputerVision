"""
practice with the re_helper file
"""
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import pyrealsense2 as rs
import rs_help as h
import sys


def main():
    try:
        myRS = h.MyRealsense()
    except:
        sys.exit('COULD NOT CREATE REALSENSE OBJECT')

    # configure object
    # not needed because these values are the same as the defaults
    # this is just to show how to do it
    # these parameters must be changed before the realsense is configured
    try:
        myRS.rs_width = 640
        myRS.rs_height = 480
        myRS.rs_frame_rate = 30
    except:
        sys.exit('COULD NOT UPDATE CLASS CONFIGURATIONS')

    # configure
    if not myRS.rs_configure():
        sys.exit('COULD NOT CONFIGURE REALSENSE')

    # start pipeling
    if not myRS.rs_start_pipe():
        sys.exit('COULD NOT START PIPELINE')

    # wait for shutter lense
    if not myRS.rs_lens_buffer(5):
        myRS.rs_stop_pipe()
        sys.exit('COULD NOT WAIT FOR LENSE TO CONFIGURE')

    while 1:
        if not myRS.rs_align_and_update_frames():
            myRS.rs_stop_pipe()
            sys.exit('COULD NOT ALIGN NOR UPDATE FRAMES')

        cv.namedWindow('Realsense', cv.WINDOW_AUTOSIZE)
        cv.imshow('Realsense', myRS.rs_get_color())
        cv.waitKey(1)



    # if not myRS.rs_align_and_update_frames():
    #     myRS.rs_stop_pipe()
    #     sys.exit('COULD NOT ALIGN NOR UPDATE FRAMES')



    color = myRS.rs_get_color()
    # cv.imwrite('lines.png', color)
    hsv = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    plt.imshow(hsv)
    plt.show()


    # color = myRS.rs_color_frame
    # depth = myRS.rs_depth_frame
    # color_map = cv.applyColorMap(cv.convertScaleAbs(depth, alpha=0.03), cv.COLORMAP_JET)
    #
    # cv.imshow('color', color)
    # cv.imshow('color_map', color_map)
    # cv.waitKey(0)


if __name__ == '__main__':
    main()
