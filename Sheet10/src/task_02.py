import numpy as np
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def click_event(event, x, y, flags, param):

    # --- YOUR CODE HERE ---#
    # TODO extract x, y coordinates from a mouse event (click)


def extract_points(image):

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', click_event)

    pts_x, pts_y = [], []

    # --- YOUR CODE HERE ---#
    # TODO get the coordinates from a mouse event (click) for n_points and return the results in an array [n_points, 2]
    # TODO mark the selected points on the image

    return pts_xy


def compute_Perspective(pts_src, pts_target):

    # --- YOUR CODE HERE ---#
    # TODO compute Perspective transformation from a set of corresponding points
    # Do not use RANSAC here

    return


def task_02():
    # --- YOUR CODE HERE ---#
    # TODO read the image and extract the coordinates of corner points

    # --- YOUR CODE HERE ---#
    # TODO get pts_target

    # --- YOUR CODE HERE ---#
    # TODO compute the perspective transformation
    # = compute_Perspective(pts_src, pts_target)

    # --- YOUR CODE HERE ---#
    # TODO warp the image and visualize the result


if __name__ == "__main__":

    task_02()

