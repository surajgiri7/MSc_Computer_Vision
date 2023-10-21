# importing all the required libraries
import cv2 as cv
import random as random
import numpy as np

def display(window_name, img):
    image = cv.imread(img)
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Problem 1
# a. function to compute and displau the integral image without using the function integral
def compute_integral_image( img):
    image = cv.imread(img)
    intensity_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # computing the integral image using the formula
    integral_image = np.zeros(intensity_image.shape)
    for i in range(intensity_image.shape[0]): # for each row
        for j in range(intensity_image.shape[1]): # for each column
            if i == 0 and j == 0: # for the first pixel
                integral_image[i][j] = intensity_image[i][j]
            elif i == 0: # for the first row
                integral_image[i][j] = intensity_image[i][j] + integral_image[i][j-1]
            elif j == 0: # for the first column
                integral_image[i][j] = intensity_image[i][j] + integral_image[i-1][j]
            else: # for the rest of the pixels
                integral_image[i][j] = intensity_image[i][j] + integral_image[i-1][j] + integral_image[i][j-1] - integral_image[i-1][j-1]
    
    integral_image = np.array(integral_image, dtype = np.uint8)
    return integral_image

    
# b.1 function to compute the mean grey value of the image by summing up 
# each pixel value in the integral image
def compute_mean_grey_value_1(img):
    integral_image = compute_integral_image(img) # computing the integral image
    sum_grey_value = 0
    
    # summing up each pixel value in the integral image and calculating the mean grey value
    for row in range(integral_image.shape[0]): 
        for col in range(integral_image.shape[1]):
            sum_grey_value += integral_image[row][col]
    total_pixels = integral_image.shape[0] * integral_image.shape[1]
    mean_grey = sum_grey_value / total_pixels
    return mean_grey
    
if __name__ == "__main__":
    img = "bonn.png"
    
    # Problem 1 a. display the integral image
    integral_image = compute_integral_image(img)
    cv.imshow("Integral Image",integral_image)
    cv.waitKey(0)
    
    # Problem 1 b.1 display the mean grey value of the image
    mean_grey_value = compute_mean_grey_value_1(img)
    print("Mean grey value of the image is: ", mean_grey_value)
    
    
    
    
    cv.destroyAllWindows()