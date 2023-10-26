import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

###########################################################
#                                                         #
#                        TASK 1                           #
#                                                         #  
###########################################################

def get_convolution_using_fourier_transform(image, kernel):
    # TODO: implement
    raise NotImplementedError


def get_convolution(image, kernel):
    # TODO: implement
    raise NotImplementedError


def task1():
    image = cv2.imread("./data/einstein.jpeg", cv2.IMREAD_GRAYSCALE)
    kernel = None  # TODO: calculate kernel

    cv_result = None # TODO: cv2.filter2D
    conv_result = get_convolution(image, kernel)
    fft_result = get_convolution_using_fourier_transform(image, kernel)

    # TODO: compare results

###########################################################
#                                                         #
#                        TASK 2                           #
#                                                         #  
###########################################################

def normalized_cross_correlation(image, template):
    # TODO: implement
    raise NotImplementedError

def ssd(image, template):
    # TODO: implement
    raise NotImplementedError

def draw_rectangle_at_matches(image, template_h, template_w, matches):
    # TODO: implement
    raise NotImplementedError

def task2():
    image = cv2.imread("./data/lena.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/eye.png", cv2.IMREAD_GRAYSCALE)

    # convert to float and apply intensity transformation to image

    result_ncc = normalized_cross_correlation(image, template)
    result_ssd = ssd(image, template)

    # TODO: draw rectangle around found locations
    # TODO: show the results


###########################################################
#                                                         #
#                        TASK 3                           #
#                                                         #  
###########################################################


def build_gaussian_pyramid_opencv(image, num_levels):
    # TODO: implement
    raise NotImplementedError


def build_gaussian_pyramid(image, num_levels):
    # TODO: implement
    raise NotImplementedError


def template_matching_multiple_scales(pyramid_image, pyramid_template):
    # TODO: implement
    raise NotImplementedError


def task3():
    image = cv2.imread("./data/traffic.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/traffic-template.png", cv2.IMREAD_GRAYSCALE)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)

    my_pyramid = build_gaussian_pyramid(image, 4)
    my_pyramid_template = build_gaussian_pyramid(template, 4)

    # TODO: compare and print mean absolute difference at each level

    # TODO: calculate the time needed for template matching without the pyramid

    result = template_matching_multiple_scales(my_pyramid, my_pyramid_template)
    # TODO: calculate the time needed for template matching with the pyramid

    # TODO: show the template matching results using the pyramid


###########################################################
#                                                         #
#                        TASK 4                           #
#                                                         #  
###########################################################


def get_derivative_of_gaussian_kernel(size, sigma):
    # TODO: implement
    raise NotImplementedError


def non_max_suppression(gradient_magnitude, gradient_direction):
    # TODO: implement
    raise NotImplementedError


def task4():
    image = cv2.imread("./data/einstein.jpeg", cv2.IMREAD_GRAYSCALE)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(7, 2)

    edges_x = None  # TODO: convolve with kernel_x
    edges_y = None  # TODO: convolve with kernel_y

    magnitude = None  # TODO: compute edge magnitude
    direction = None  # TODO: compute edge direction
    
    suppressed_image = non_max_suppression(magnitude, direction)

    # Sobel
    sobel_kernel_x, sobel_kernel_y = None, None

    edge_sobel_x = None  # TODO: compute the edges with sobel
    edge_sobel_y = None  # TODO: compute the edges with sobel

    # TODO: compute the mean absolute error


###########################################################
#                                                         #
#                        TASK 5                           #
#                                                         #  
###########################################################


def task5():
    image = cv2.imread("./data/traffic.jpg", cv2.IMREAD_GRAYSCALE)

    edges = None  # TODO: compute edges

    dist_transfom_cv = None  # TODO: compute using opencv

    dist_transfom_cv_filtered = None  # TODO: compute after filtering some high-frequency edges


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
