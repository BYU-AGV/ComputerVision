from rs_help002 import Realsense435
import cv2 as cv
import keyboard
from map_2d import get_map
import numpy as np

if __name__ == '__main__':
    myRS = Realsense435()
    myRS.start_rs()

    while not keyboard.is_pressed('space'):
        color, depth = myRS.get_frames()

        myMap, mask, clr = get_map(color, depth)

        myMap = np.flip(myMap)

        color = cv.cvtColor(color, cv.COLOR_RGB2BGR)
        clr = cv.cvtColor(clr, cv.COLOR_RGB2BGR)

        cv.imshow('color', color)
        cv.imshow('depth', myMap)
        cv.imshow('mask', mask)
        cv.imshow('cut color', clr)
        cv.waitKey(1)

    cv.destroyAllWindows()
    myRS.stop_pipe()
    print('Done!')