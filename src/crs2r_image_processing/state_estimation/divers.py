import cv2 as cv


def init_win_subimages():
    cs = [
        (450, 330),
        (650, 330),
        (650, 100),
        (450, 100),
        (550, 200),
    ]  # ul corners coordinates of plate corners subimages windows
    for i, c in enumerate(cs):
        cv.namedWindow("sub_" + str(i))
        cv.moveWindow("sub_" + str(i), c[0] - 200, c[1] + 500)
