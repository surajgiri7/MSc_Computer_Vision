import cv2 as cv
import random as random
import numpy as np
import matplotlib.pyplot as plt 

def display(window_name, img):
    image = cv.imread(img)
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def detect_lines(image_path):
    # Read the image
    image = cv.imread(image_path)
    
    # Convert the image to grayscale
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv.Canny(gray_img, 50, 150, apertureSize=3)
    
    # Perform Hough Line Transform
    lines = cv.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    # Draw the detected lines on the original image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save the result image
    cv.imwrite('detected_lines.png', image)


'''
diagram to understand hough transform 
    the line is 90 degree to the origin
# -------------------------
#   *
#  *  (rho)
# *
#*theta
################
so we require a accumaltor array to store all the intersection


'''

def myhoughLines(image, rho_res, theta_res, threshold):
    max_rho = int(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))
    possible_rho_values = np.arange(-max_rho, max_rho + 1, rho_res)
    possible_theta_values = np.deg2rad(np.arange(0, 180, theta_res))

    # Create meshgrid for vectorized computation
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    # Calculate rho values for all theta values and all edge points
    rho_values = np.outer(np.sin(possible_theta_values), x.flatten()) + np.outer(np.cos(possible_theta_values), y.flatten())
    rho_values = rho_values.astype(int).reshape((len(possible_theta_values), image.shape[0], image.shape[1]))

    # Find the nearest index for each rho value in possible_rho_values
    rho_indices = np.argmin(np.abs(possible_rho_values[:, None] - rho_values), axis=0)

    # Accumulate votes in the accumulator array
    acc = np.zeros((len(possible_rho_values), len(possible_theta_values)), dtype=np.uint64)
    acc[rho_indices, np.arange(len(possible_theta_values))] += (image > 0).flatten()

    # Identify peaks in the accumulator
    detected_lines = np.argwhere(acc > threshold)

    # Convert peaks to lines
    lines = [(possible_rho_values[rho_index], possible_theta_values[theta_index]) for rho_index, theta_index in detected_lines]

    return acc, lines


def visualize_hough(image, accumulator, lines):
    # Visualize the accumulator
    cv.imshow('Hough Accumulator', accumulator)

    # Visualize the original image with detected lines
    image_with_lines = cv.cvtColor(image.copy(), cv.COLOR_GRAY2RGB)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho + 1000 * (-b))
        y0 = int(b * rho + 1000 * (a))
        x1 = int(a * rho - 1000 * (-b))
        y1 = int(b * rho - 1000 * (a))
        cv.line(image_with_lines, (x0, y0), (x1, y1), (0, 0, 255), 2)

    cv.imshow('Image with Detected Lines', image_with_lines)
    cv.waitKey(0)
    cv.destroyAllWindows()


#detect_lines('images/shapes.png')


# Example usage:
image_path = 'images/shapes.png'
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
rho_resolution = 1
theta_resolution = np.pi / 180
threshold = 100

accumulator, detected_lines = myhoughLines(image, rho_resolution, theta_resolution, threshold)
visualize_hough(image, accumulator, detected_lines)



