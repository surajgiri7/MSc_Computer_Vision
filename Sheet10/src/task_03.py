import numpy as np
import cv2
import random

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_Homography_RANSAC(good_matches, kp_1, kp_2, image_1, image_2):

    # --- YOUR CODE HERE ---#
    # TODO compute best H transformation using RANSAC algorithm

    return best_H


def get_best_matches(des_1, des_2, thr=0.4):

    # --- YOUR CODE HERE ---#
    # TODO get best matches

    return good_matches

def task_03():

    # --- YOUR CODE HERE ---#
    # TODO read the image and extract key-points

    image_1 = cv2.imread(r'./data/Mountain_1.png')
    image_2 = cv2.imread(r'./data/Mountain_2.png')

    # --- YOUR CODE HERE ---#
    # TODO get the best matching
    good_matches = get_best_matches(des_1, des_2)

    # --- YOUR CODE HERE ---#
    # TODO visualize good matches

    # --- YOUR CODE HERE ---#
    # TODO implement RANSAC algorithm to compute the best transformation H
    H = compute_Homography_RANSAC(good_matches, kp_1, kp_2, image_1, image_2)

    # --- YOUR CODE HERE ---#
    # TODO Stitch the images and visualize the result


if __name__ == "__main__":

    task_03()

