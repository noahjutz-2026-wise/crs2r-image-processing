import numpy as np
import cv2 as cv


def mask_hsv(img, color_params=None):
    imageHsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    minHsv = np.array([color_params[0][0], color_params[1][0], color_params[2][0]])
    maxHsv = np.array([color_params[0][1], color_params[1][1], color_params[2][1]])

    mask = cv.inRange(imageHsv, minHsv, maxHsv)
    return None, mask
