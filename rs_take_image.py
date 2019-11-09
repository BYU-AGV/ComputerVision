"""
This file will take and save pictures using the intel Realsense
"""
import rs_help
import cv2 as cv
import numpy as np

file_name = 'lines.png'

if __name__ == '__main__':
    # create realsense object
    myRS = rs_help.MyRealsense()
    # config
    myRS.rs_configure()
    # start pipe
    myRS.rs_start_pipe()
    # buffer lens
    myRS.rs_lens_buffer(20)
    # update frames
    myRS.rs_align_and_update_frames()
    # grab image
    my_image = myRS.rs_get_color()
    # close pipe
    myRS.rs_stop_pipe()
    # save image
    cv.imwrite(file_name, my_image)

    print('photo taken')
