import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

###########################################################
#                                                         #
#                        TASK 1                           #
#                                                         #  
###########################################################


def display(window_name, img):

    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






def get_convolution_using_fourier_transform(image, kernel):

    # Calculate the Fourier transforms of the image and kernel
    ft_img = np.fft.fft2(image)
    ft_kernel = np.fft.fft2(kernel, s=image.shape)

    # Perform frequency domain multiplication
    ft_result = ft_img * ft_kernel

    # Transform the result back to the spatial domain
    result = np.fft.ifft2(ft_result).real  # .real extracts the real part

    # Normalize and convert the result to uint8
    result = (result - result.min()) / (result.max() - result.min()) * 255
    return result.astype(np.uint8)



def get_convolution_using_fourier_transform_correction(image, kernel):
    # Calculate the Fourier transforms of the image and kernel
    ft_image = np.fft.fft2(image)
    ft_kernel = np.fft.fft2(kernel, s=image.shape)

    # Perform frequency domain multiplication
    ft_result = ft_image * ft_kernel

    # Transform the result back to the spatial domain
    result = np.fft.ifft2(ft_result).real  # .real extracts the real part

    # Ensure the result is within the valid data range (0-255)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def convolution_spatial_vectorized(image, kernel):
    k_height, k_width = kernel.shape
    k_center_x, k_center_y = k_width // 2, k_height // 2

    # Pad the image to handle borders
    padded_image = np.pad(image, ((k_center_y, k_center_y), (k_center_x, k_center_x)), mode='constant')

    # Perform convolution using vectorized operations
    result = np.zeros_like(image, dtype=np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result[y, x] = np.sum(padded_image[y:y + k_height, x:x + k_width] * kernel)

    # Normalize the result to be within the 0-255 range
    result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255

    return result.astype(np.uint8)

def get_convolution(image, kernel): 

    result = cv2.filter2D(image, -1, kernel) 
    return result
    


def task1():
    image = cv2.imread("./data/einstein.jpeg", cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getGaussianKernel(7,1)  
    gaussian_kernel = kernel.dot(kernel.reshape(1, 7)) 
    conv_result = get_convolution(image, gaussian_kernel)  
    own_conv = convolution_spatial_vectorized(image,kernel)
    fft_result = get_convolution_using_fourier_transform(image,kernel) 
    fft_result_correction = get_convolution_using_fourier_transform_correction(image,kernel)
    display("original image",image) 
    display("convolution image",conv_result) 
    display("own function",own_conv)
    display("fft function",fft_result) 
    #oscillations or ripples around high-contrast edges in the image in fft. 
    display("fft_result_correction",fft_result_correction)
    mean_abs_diff = np.mean(np.abs(np.subtract(conv_result.astype(np.int32), fft_result.astype(np.int32))))
    print(mean_abs_diff)
   
     
   

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

    edges = cv2.Canny(image, 50, 60)

    display("edges",edges) 

    dist_transfom_cv = cv2.distanceTransform(image, cv2.DIST_L2, 0)

    display("distance_cv",dist_transfom_cv)

    dist_transfom_cv_filtered = None  # TODO: compute after filtering some high-frequency edges


if __name__ == "__main__":
    #task1()
    #task2()
    #task3()
    #task4()
    task5()
