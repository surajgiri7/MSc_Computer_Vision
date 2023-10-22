# importing all the required libraries
import cv2 as cv
import random as random
import numpy as np
import matplotlib.pyplot as plt

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

# b.2 function to compute the integral image using the function integral
def compute_integral_image_function(img):
    image = cv.imread(img)
    intensity_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    integral_image = cv.integral(intensity_image)
    integral_image = np.array(integral_image, dtype = np.uint8)
    return integral_image

# b.3 function to compute the integral image using my own function
def compute_integral_image_own_func(img):
    image = cv.imread(img)
    myfunc_image = compute_integral_image(img)
    return myfunc_image


##2a histogram equlaization 

def histogram(img): 
    hist_img = cv.equalizeHist(img) 
    cv.imwrite('bonn_equalized.png', hist_img)
    # Display the original and equalized images (optional)
    return hist_img

def own_histogram(img):
    
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Apply histogram equalization
    equalized_image = np.interp(img, np.arange(256), cdf_normalized)
    equalized_image = np.uint8(equalized_image)
    
    return equalized_image 

# 5. function to read the image and convert it to grayscale
def read_image_and_to_grey(img):
    image = cv.imread(img)
    intensity_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return intensity_image

# 5.a function to filter the image twice with a Gaussian Kernel with sigma = 2
def filter_twice_gaussian(img):
    intensity_image = read_image_and_to_grey(img)
    sigma = 2
    # applying the Gaussian kernel with sigma = 2 twice
    for _ in range(2):
        filtered_image = cv.GaussianBlur(intensity_image, (5,5), sigma)
    return filtered_image

# 5.b function to filter the image once with a Gaussian Kernel with sigma = 2
def filter_with_gaussian(img):
    intensity_image = read_image_and_to_grey(img)
    sigma = 2 * np.sqrt(2)
    # applying the Gaussian kernel with sigma = 2 * sqrt(2)
    filtered_image = cv.GaussianBlur(intensity_image, (5,5), sigma)
    return filtered_image

# 5. computing the absolute pixelwise difference between the two filtered images and 
# printing the maximum pixel error
def compute_pixelwise_difference(img):
    twice_filtered_image = filter_twice_gaussian(img)
    once_filtered_image = filter_with_gaussian(img)
    difference_image = np.abs(twice_filtered_image - once_filtered_image)
    max_pixel_error = np.max(difference_image)
    return max_pixel_error

# 7. function to add 30% salt and pepper noise to the the image
def add_salt_and_pepper_noise(img):
    grayscale_image = read_image_and_to_grey(img)
    noise_image = np.zeros(grayscale_image.shape, dtype = np.uint8)
    salt_pepper = 0.3
    for row in range(grayscale_image.shape[0]):
        for col in range(grayscale_image.shape[1]):
            random_number = random.random()
            if random_number < salt_pepper / 2:
                noise_image[row][col] = 0
            elif random_number < salt_pepper:
                noise_image[row][col] = 255
            else:
                noise_image[row][col] = grayscale_image[row][col]
    return noise_image

# 7.a function to filter the image with a Gaussian Kernel
def filter_salt_n_pepper_with_gaussian(img):
    salt_pepper_image = add_salt_and_pepper_noise(img)
    sigma = 5
    # applying the Gaussian kernel with sigma = 3
    filtered_image = cv.GaussianBlur(salt_pepper_image, (5,5), sigma)
    return filtered_image

# 7.b function to filter the image with a median filter
def filter_salt_n_pepper_with_median(img):
    salt_pepper_image = add_salt_and_pepper_noise(img)
    # applying the median filter
    filtered_image = cv.medianBlur(salt_pepper_image, 5)
    return filtered_image

# 7.c function to filter the image with a bilateral filter
def filter_salt_n_pepper_with_bilateral(img):
    salt_pepper_image = add_salt_and_pepper_noise(img)
    # applying the bilateral filter
    filtered_image = cv.bilateralFilter(salt_pepper_image, 5, 75, 75)
    return filtered_image


    
if __name__ == "__main__":
    img = "bonn.png"
    
    # Problem 1 a. display the integral image
    integral_image = compute_integral_image(img)
    cv.imshow("Integral Image",integral_image)
    cv.waitKey(0)
    
    # Problem 1 b.1 display the mean grey value of the image
    mean_grey_value = compute_mean_grey_value_1(img)
    print("Mean grey value of the image is: ", mean_grey_value)
    
    # Problem 1 b.2 display the integral image using the function integral
    integral_image_function = compute_integral_image_function(img)
    cv.imshow("1.b.2 Integral Image using Function - integral",integral_image_function)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Problem 1 b.3 display the integral image using my own function
    integral_image_own_func = compute_integral_image_own_func(img)
    cv.imshow("1.b.3 Integral Image using Own Function",integral_image_own_func)
    cv.waitKey(0)
    cv.destroyAllWindows()

    image = cv.imread(img)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    pixel_lib = histogram(gray_image) 
    custom_lib = own_histogram(gray_image) 
    pixel_diff = np.abs(pixel_lib - custom_lib)
    # Find the maximum pixel error
    max_pixel_error = np.max(pixel_diff) 
    cv.imshow('Equalized Image (OpenCV)', pixel_lib)
    cv.imshow('Equalized Image (Custom)', custom_lib)
    cv.imshow('Pixelwise Difference', pixel_diff)
    print(f"Maximum Pixel Error: {max_pixel_error}")
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Problem 5 - display the grayscale image
    grey_image = read_image_and_to_grey(img)
    cv.imshow("5. Grayscale Image", grey_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Problem 5.a - display the image filtered twice with a Gaussian Kernel with sigma = 2
    filter_twice_image_gaussian = filter_twice_gaussian(img)
    cv.imshow("5.a Twice filtered Image with Gaussian kernel", filter_twice_image_gaussian)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Problem 5.b - display the image filtered once with a Gaussian Kernel with sigma = 2 * sqrt(2)
    filter_once_image_gaussian = filter_with_gaussian(img)
    cv.imshow("5.b Once filtered Image with Gaussian kernel", filter_once_image_gaussian)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Problem 5 - computing and printing the absolute pixelwise difference between the two filtered images
    max_pixel_error = compute_pixelwise_difference(img)
    print("5. Maximum pixel error is: ", max_pixel_error)
    
    # Problem 7 - display the image with 30% salt and pepper noise
    noise_image = add_salt_and_pepper_noise(img)
    cv.imshow("7. Image with 30% salt and pepper noise", noise_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Problem 7.a - display the image filtered with a Gaussian Kernel
    gaussian_filtered_salt_n_pepper = filter_salt_n_pepper_with_gaussian(img)
    cv.imshow("7.a Image filtered with Gaussian kernel", gaussian_filtered_salt_n_pepper)
    cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # Problem 7.b - display the image filtered with a median filter
    median_filtered_salt_n_pepper = filter_salt_n_pepper_with_median(img)
    cv.imshow("7.b Image filtered with median filter", median_filtered_salt_n_pepper)
    cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # Problem 7.c - display the image filtered with a bilateral filter
    bilateral_filtered_salt_n_pepper = filter_salt_n_pepper_with_bilateral(img)
    cv.imshow("7.c Image filtered with bilateral filter", bilateral_filtered_salt_n_pepper)
    cv.waitKey(0)
    
    cv.imshow("Filter Comparison", np.vstack((gaussian_filtered_salt_n_pepper, median_filtered_salt_n_pepper, bilateral_filtered_salt_n_pepper)))
    cv.waitKey(0)
    

    
    
    
    
    cv.destroyAllWindows()