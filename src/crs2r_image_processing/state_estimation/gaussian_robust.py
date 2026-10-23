import cv2 as cv
import numpy as np


def detect_gaussian(mask, j, q, th, show_sub=False):
    if j < 4:  # corners
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        if len(contours) == 0:
            c = (np.asarray(mask.shape) - 1.0) / 2.0
            blob_found = False
        else:
            contour = max(contours, key=cv.contourArea)
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                c = np.array([cy, cx])
                blob_found = True
            else:
                c = (np.asarray(mask.shape) - 1.0) / 2.0
                blob_found = False

    else:  # ball
        kernel_erosion = np.ones((2, 2), np.uint8)
        kernel_dilatation = np.ones((5, 5), np.uint8)
        erosion = cv.erode(mask, kernel_erosion, iterations=1)
        mask = cv.dilate(erosion, kernel_dilatation, iterations=1)

        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        if len(contours) == 0:
            c = np.asarray(mask.shape) / 2.0
            blob_found = False
        else:
            contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, True)
            if perimeter == 0:
                circularity = 0
            else:
                circularity = (4 * np.pi * area) / (perimeter ** 2)

            M = cv.moments(contour)

            if M["m00"] != 0 and circularity > 0.12 and area > 110:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                c = np.array([cy, cx])
                blob_found = True
            else:
                c = np.asarray(mask.shape) / 2.0
                blob_found = False

    if show_sub:
        cv.imshow("sub_{}".format(j), mask)

    return c, blob_found
