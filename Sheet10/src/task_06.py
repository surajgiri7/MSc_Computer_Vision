import matplotlib.pyplot as plt
import numpy as np
import cv2

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def draw_Epipolar(im1, im2, corr1, corr2, F):

    # --- YOUR CODE HERE ---#
    # TODO draw and visualize the corresponding points and Epipolar lines

def compute_fundamintal_matrix(corr1, corr2, is_norm=True):

    # --- YOUR CODE HERE ---#
    # TODO compute the fundamental matrix F using the eight-points algorithm

    return F


def task_06():

    # --- YOUR CODE HERE ---#
    # TODO read the images and the coordinates

    im_1 = cv2.imread(r'./data/Uni_Bonn_01.JPG')
    im_2 = cv2.imread(r'./data/Uni_Bonn_02.JPG')

    corr_all = np.genfromtxt(r'./data/corr.txt', dtype=float)
    corr_1 = corr_all[:, :2]
    corr_2 = corr_all[:, 2:]

    # --- YOUR CODE HERE ---#
    # TODO compute the fundamental matrix F using the eight-points algorithm

    F = compute_fundamintal_matrix(corr_1, corr_2)

    # --- YOUR CODE HERE ---#
    # TODO check the result of computing F and print the results

    # --- YOUR CODE HERE ---#
    # TODO draw and visualize the corresponding points and the draw the Epipolar lines

    draw_Epipolar(im_1, im_2, corr_1, corr_2, F)

    # --- YOUR CODE HERE ---#
    # TODO rectify the images and visualize them with the corresponding points and Epipolar lines





if __name__ == "__main__":

   task_06()

