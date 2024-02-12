import numpy as np
import cv2

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
    # TODO mark/plot the selected points with IDs on the image

    return pts_xy



def compute_Homography(pts_src, pts_target):

    # --- YOUR CODE HERE ---#
    # TODO compute Homography from a set of corresponding points
    # Do not use RANSAC here

    return H


def check_Homography(pts_src, pts_target, H):

    # --- YOUR CODE HERE ---#
    # TODO check the Homography transformation and print the differences




def task_01():

    # --- YOUR CODE HERE ---#
    # TODO read the images and extract coordinates of corresponding points

    #image_src =
    #image_target =
    pts_src = extract_points(image_src.copy())
    pts_target = extract_points(image_target.copy())

    # --- YOUR CODE HERE ---#
    # TODO cmpute homography and check it
    H = compute_Homography(pts_src, pts_target)

    check_Homography(pts_src, pts_target, H)

    # --- YOUR CODE HERE ---#
    # TODO stitch the images and visualize the result


if __name__ == "__main__":

   task_01()

