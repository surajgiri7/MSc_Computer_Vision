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
    display("1. original image",image) 
    display("1. convolution image",conv_result) 
    display("1. own function",own_conv)
    display("1. fft function",fft_result) 
    #oscillations or ripples around high-contrast edges in the image in fft. 
    display("1. fft_result_correction",fft_result_correction)
    mean_abs_diff = np.mean(np.abs(np.subtract(conv_result.astype(np.int32), fft_result.astype(np.int32))))
    print("mean Absolute Difference: ", mean_abs_diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
     


###########################################################
#                                                         #
#                        TASK 2                           #
#                                                         #  
###########################################################

def normalized_cross_correlation(image, template):
    k, l = template.shape
    n_rows = image.shape[0] - k + 1 
    n_cols = image.shape[1] - l + 1

    normalized_cross_correlation = np.zeros((n_rows, n_cols))

    xcorr_template = template - np.mean(template)
    sum_norm_template = np.sum(xcorr_template ** 2)

    for row in range(n_rows):
        for col in range(n_cols):
            image_patch = image[row:row + k, col:col + l]
            xcorr_image = image_patch - np.mean(image_patch)

            normalized_cross_correlation[row, col] = np.sum(xcorr_template * xcorr_image) / np.sqrt(sum_norm_template * np.sum(xcorr_image ** 2))

    return normalized_cross_correlation


    

def ssd(image, template): # measuring normalized square sum difference
    k, l = template.shape
    n_rows = image.shape[0] - k + 1
    n_cols = image.shape[1] - l + 1

    ssd = np.zeros((n_rows, n_cols))

    for row in range(n_rows):
        for col in range(n_cols):
            image_patch = image[row:row + k, col:col + l]
            ssd[row, col] = np.sum((template - image_patch) ** 2)

    return ssd 



def draw_rectangle_at_matches(image, template_h, template_w, matches):
    image_copy = image.copy()

    for row in range(matches.shape[0]):
        for col in range(matches.shape[1]):
            if matches[row, col]:
                cv2.rectangle(image_copy, (col, row), (col + template_w, row + template_h), (0,0, 255), 1)

    return image_copy


def task2():
    image = cv2.imread("./data/lena.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/eye.png", cv2.IMREAD_GRAYSCALE)

    # convert to float and apply intensity transformation to image and template
    result_ncc = normalized_cross_correlation(image, template)
    result_ssd = ssd(image, template)

    # drawing rectangles around matches where similarity <= 0.1 for SSD and >= 0.7 for NCC using np.where
    matches_ncc = np.where(result_ncc >= 0.7, 1, 0) 
    matches_ssd = np.where(result_ssd <= 0.1, 1, 0)

    # drawing rectangles around matches where similarity <= 0.1 for SSD and >= 0.7 for NCC using np.where
    image_ncc = draw_rectangle_at_matches(image, template.shape[0], template.shape[1], matches_ncc)
    image_ssd = draw_rectangle_at_matches(image, template.shape[0], template.shape[1], matches_ssd)

    # display results
    cv2.imshow("2. NCC Result", image_ncc)
    cv2.waitKey(0)
    cv2.imshow("2. SSD Result", image_ssd)
    cv2.waitKey(0)

    # subtracting 0.5 to the image, making sure the values do not become negative,
    #  and repeating the template matching 
    image = image.astype(np.float32) - 0.5
    image[image < 0] = 0
    result_ncc = normalized_cross_correlation(image, template)
    result_ssd = ssd(image, template)

    # drawing rectangles around matches where similarity <= 0.1 for SSD and >= 0.7 for NCC using np.where
    matches_ncc = np.where(result_ncc >= 0.7, 1, 0)
    matches_ssd = np.where(result_ssd <= 0.1, 1, 0)

    # drawing rectangles around matches where similarity <= 0.1 for SSD and >= 0.7 for NCC using np.where
    new_image_ncc = draw_rectangle_at_matches(image, template.shape[0], template.shape[1], matches_ncc)
    new_image_ssd = draw_rectangle_at_matches(image, template.shape[0], template.shape[1], matches_ssd)

    # display results
    cv2.imshow("2. NCC Result - 0.5", new_image_ncc)
    cv2.waitKey(0)
    cv2.imshow("2. SSD Result - 0.5", new_image_ssd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   

    """
    When I performed the subtraction of 0.5 to the image, making sure the values do not become negative,
    and repeating the template matching, the results were not as good as the first one.
    The reason is might be that the image is not normalized anymore, so the values are 
    not in the range [0,1].
    """



###########################################################
#                                                         #
#                        TASK 3                           #
#                                                         #  
###########################################################


def build_gaussian_pyramid_opencv(image, num_levels):
    
    pyramid = [image]
    for _ in range(num_levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def custom_pyrDown(image):
    kernel = cv2.getGaussianKernel(3, 1)
    kernel = kernel.dot(kernel.T)
    # Convolve the image with the Gaussian kernel
    blurred = cv2.filter2D(image, -1, kernel)
    # Down-sample by selecting every second pixel
    downsampled = blurred[::2, ::2]

    return downsampled

def build_gaussian_pyramid(image, num_levels):
    
    """ 
    own function for gaussian pyramid
    """
    pyramid = [image]
    for _ in range(num_levels):
        image = custom_pyrDown(image)
        pyramid.append(image)
    return pyramid 


def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
    results = []
    level = len(pyramid_image)
    print("# of Level:", level)

    for idx in range(0, level):
        refimg = pyramid_image[-idx]
        tplimg = pyramid_template[-idx]

        # Perform template matching
        result = cv2.matchTemplate(refimg, tplimg, cv2.TM_CCORR_NORMED)

        if idx > 0:
            # Calculate mask for the pyramid level
            mask = cv2.pyrUp(results[-1])
            T, mask = cv2.threshold(mask, threshold, 1.0, cv2.THRESH_BINARY)

            # Use binary thresholding to convert to 0s and 1s
            mask = (mask > 0).astype(np.float32)

            # Create a mask that reflects the size of the template
            mask_template = cv2.matchTemplate(refimg, tplimg, cv2.TM_CCORR_NORMED)
            mask_template = (mask_template > threshold).astype(np.float32)

            if mask.shape != result.shape:
                mask = cv2.resize(mask, (result.shape[1], result.shape[0]))

            # Apply the masks
            mask *= mask_template
            result *= mask

        results.append(result)

    return results



def task3():
    # Load the image and template
    image = cv2.imread("./data/traffic.jpg", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("./data/traffic-template.png", cv2.IMREAD_GRAYSCALE)

    my_pyramid = build_gaussian_pyramid(image, 4)
    cv_pyramid = build_gaussian_pyramid_opencv(image, 4)

    for x, (cv_img, custom_img) in enumerate(zip(cv_pyramid, my_pyramid)):
        diff = np.mean(np.abs(cv_img - custom_img))
        print(f'3. Mean abs difference: level {x}: {diff:.2f}')

    start_time = time.time()
    result = normalized_cross_correlation(image, template)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"3. Time taken for NCC: {elapsed_time:.4f} seconds")

    pyramid_template = build_gaussian_pyramid(template, 4)
    start_time_pyramid = time.time()
    result_pyramid = template_matching_multiple_scales(my_pyramid, pyramid_template, 0.5)
    end_time_pyramid = time.time()
    elapsed_time_pyramid = end_time_pyramid - start_time_pyramid
    print(f"3. Time taken for NCC with pyramid: {elapsed_time_pyramid:.4f} seconds")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


###########################################################
#                                                         #
#                        TASK 4                           #
#                                                         #  
###########################################################


def get_derivative_of_gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel_2D = kernel.dot(kernel.reshape(1, size)) 
    dx, dy = np.gradient(kernel_2D)
    print("4. Weights of derivative of gaussian kernel in X direction")
    print(dx)
    print("4. Weights of derivative of gaussian kernel in Y direction")
    print(dy)
    return dx, dy

def non_max_suppression(gradient_magnitude, gradient_direction):
    m, n = gradient_magnitude.shape
    resulting_image = np.zeros((m, n), dtype=np.float32)
    angle = gradient_direction * 180.
    angle[angle < 0] += 180

    for i in range (1, m-1):
        for j in range (1, n-1):
            q = 255
            r = 255

            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = gradient_magnitude[i+1, j-1]
                r = gradient_magnitude[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = gradient_magnitude[i-1, j-1]
                r = gradient_magnitude[i+1, j+1]

            if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                resulting_image[i,j] = gradient_magnitude[i,j]
            else:
                resulting_image[i,j] = 0

    return resulting_image



def thresholding(image, low, high):
    m, n = image.shape
    result = np.zeros((m,n), dtype=np.float32)

    for i in range(1, m-1):
        for j in range(1, n-1):
            if image[i,j] > high:
                result[i,j] = 255
            elif image[i,j] < low:
                result[i,j] = 0
            else:
                result[i,j] = image[i,j]
    return result


def task4():
    image = cv2.imread("./data/einstein.jpeg", cv2.IMREAD_GRAYSCALE)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(7, 2)

    # convolve the image with Gaussian kernels
    edges_x = cv2.filter2D(image.astype(np.float32), -1, kernel_x)
    edges_y = cv2.filter2D(image.astype(np.float32), -1, kernel_y)

    cv2.imshow("4. Original Image", image)
    cv2.waitKey(0)
    cv2.imshow("4. Convolving the image with dx", edges_x)
    cv2.waitKey(0)
    cv2.imshow("4. Convolving the image with dy", edges_y)
    cv2.waitKey(0)

    # computing the edge magnitude
    magnitude = np.sqrt(edges_x ** 2 + edges_y ** 2)
    # computing the edge direction
    direction = np.arctan2(edges_y, edges_x)
    print("4. Magnitude: ", magnitude)
    print("4. Direction: ", direction)
    cv2.imshow("4. Magnitude Image", magnitude)
    cv2.waitKey(0)
    cv2.imshow("4. Direction Image", direction)
    cv2.waitKey(0)
    
    suppressed_image = non_max_suppression(magnitude, direction)
    print("4. Suppressed Image: ", suppressed_image)
    cv2.imshow("4. Suppressed Image", suppressed_image)
    cv2.waitKey(0)

    # applying thresholding
    low = 0.1 * np.max(suppressed_image)
    high = 0.9 * np.max(suppressed_image)
    print("4. Low Threshold: ", low)
    print("4. High Threshold: ", high)
    threshold_image = thresholding(suppressed_image, low, high)
    print("4. Threshold Image: ", threshold_image)
    cv2.imshow("4. Thresholded Image", threshold_image)
    cv2.waitKey(0)

    # Sobel
    kernel_size = 3
    sobel_kernel_x, sobel_kernel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size), cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

    edge_sobel_x = cv2.convertScaleAbs(sobel_kernel_x)
    edge_sobel_y = cv2.convertScaleAbs(sobel_kernel_y)
    magnitude_sobel = np.sqrt(edge_sobel_x ** 2 + edge_sobel_y ** 2)

    cv2.imshow("4. Sobel Image X_direction", edge_sobel_x)
    cv2.waitKey(0)
    cv2.imshow("4. Sobel Image Y_direction", edge_sobel_y)
    cv2.waitKey(0)
    sobel_edge_detected = cv2.addWeighted(edge_sobel_x, 0.5, edge_sobel_y, 0.5, 0)
    cv2.imshow("4. Sobel Edge Detected Image", sobel_edge_detected)
    cv2.waitKey(0)

    # computing the mean absolute error
    mean_abs_error = np.mean(np.abs(np.subtract(sobel_edge_detected.astype(np.int32), threshold_image.astype(np.int32))))
    print("4. Mean Absolute Error: ", mean_abs_error)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


###########################################################
#                                                         #
#                        TASK 5                           #
#                                                         #  
###########################################################


def task5():
    image = cv2.imread("./data/traffic.jpg", cv2.IMREAD_GRAYSCALE)

    edges = cv2.Canny(image, 50, 60)

    display("5. edges",edges) 

    dist_transfom_cv = cv2.distanceTransform(image, cv2.DIST_L2, 0.7)

    display("5. distance_cv",dist_transfom_cv)




if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
