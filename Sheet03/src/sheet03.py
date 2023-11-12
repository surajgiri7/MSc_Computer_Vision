import numpy as np
import cv2 as cv
import random 
import matplotlib.pyplot as plt 


##############################################
#     Task 1        ##########################
##############################################


def display(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()



def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    image = cv.imread(img)
    
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

def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    distSteps = int(np.linalg.norm(img_edges.shape) / d_resolution)
    degSteps = int(np.pi / theta_step_sz)

    accumulator = np.zeros((degSteps, distSteps))
    with np.nditer(img_edges, flags=['multi_index']) as it:
        for px in it:
            if px > 3:
                for deg in range(degSteps):
                    p = (it.multi_index[0] * np.cos(deg)) + (it.multi_index[1] * np.sin(deg))
                    if(-distSteps/2 <= p and p < distSteps/2):
                        accumulator[deg, int(p + distSteps/2) ] += 1

    detected_lines = []
    with np.nditer(accumulator, flags=['multi_index']) as it2:
        for votes in it2:
            if votes >= threshold:
                detected_lines.append(it2.multi_index)

    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    edges = cv.Canny(img,50,150,apertureSize = 3)
    houghLines, accumulator= myHoughLines(edges,0.7,np.pi/180,55)

    display(accumulator/np.max(accumulator))

    for theta,rho in houghLines:
        rho = rho - accumulator.shape[1] / 2
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    display(img)


##############################################
#     Task 2        ##########################
##############################################

# Please provide your theoretical solution in a pdf file. 

##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    # Initialize centers using some random points from data
    centers = data[np.random.choice(data.shape[0], k, replace=False)]

    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    convergence = False
    iterationNo = 0
    while not convergence:
        # Assign each point to the cluster of the closest center
        distances = np.linalg.norm(data[:, None] - centers, axis=2)
        index = np.argmin(distances, axis=1)

        # Update clusters' centers and check for convergence
        new_centers = np.array([np.mean(data[index == j], axis=0) for j in range(k)])
        convergence = np.all(centers == new_centers)
        centers = new_centers

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers

def segment_image(image, features, k_values):
    """
    Segment the image using the myKmeans function for different feature spaces and k values
    :param image: input image
    :param features: feature space (e.g., 'intensity', 'color', 'intensity_position', 'custom')
    :param k_values: list of k values to try
    """
    if features == 'intensity':
        data = image.reshape((-1, 1))
    elif features == 'color':
        data = image.reshape((-1, image.shape[2]))
    elif features == 'intensity_position':
        height, width = image.shape[:2]
        positions = np.column_stack(np.meshgrid(np.arange(height), np.arange(width))).reshape((-1, 2))
        data = np.column_stack([image.reshape((-1, image.shape[2])), positions])
    else:
        raise ValueError("Invalid Feature space")

    for k in k_values:
        index, centers = myKmeans(data, k)

        # Visualize the results
        segmented_image = centers[index].reshape(image.shape)

        plt.figure()
        plt.title(f'Segmentation with k={k} and {features} features')
        plt.imshow(segmented_image.astype(np.uint8))
        plt.show()

def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    segment_image(img, 'intensity', [2, 4, 6])
    '''
    ...
    your code ...
    ...
    '''


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    segment_image(img, 'color', [2, 4, 6])
    '''
    ...
    your code ...
    ...
    '''


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''
    data = img.copy()
    # Create new Dataset and expand every pixel from RGB to RGBXY
    data = np.zeros((img.shape[0] * img.shape[1], 5))
    # Set RGB Values for every data point
    data[:,0:3] = img.reshape(img.shape[0] * img.shape[1], 3)

    # Set X,Y Coordinates for every data point
    for px_index, px_val in enumerate(data):
        data[px_index,3:5] = np.array([px_index // img.shape[1] + 1, px_index % img.shape[1] + 1])
        # Scale Image Coordinates to RGB Interval [0,255]
        # Also keep the x:y ratio
        data[px_index,3] = data[px_index,3] / np.maximum(img.shape[1], img.shape[0]) * 255
        data[px_index,4] = data[px_index,4] / np.maximum(img.shape[1], img.shape[0]) * 255

    for k in [2,4,6]:
        # Calculate kmeans for k = 2,4,6
        kmeans = myKmeans(data, k)
        indices = kmeans[0]
        img_copy = img.reshape(img.shape[0] * img.shape[1], 3)
        # Iterate over clusters and replace pixels with Cluster Value
        for cluster_id, cluster in enumerate(kmeans[1]):
            cluster_indices = np.where(indices == cluster_id)
            img_copy[cluster_indices] = cluster[0:3]



def task_3_d():
    print("Task 3 (d) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''

##############################################
#     Task 4        ##########################
##############################################

def meanShift(data, window_size, kernel, x1, y1):
    """
    Your implementation of mean shift algorithm
    :param data: data points to cluster
    :param window_size: the size of the window is 2 * window_size + 1 
    :param kernel: the chosen kernel (slide 74)
    :param x1, y1: original position (x1, y1)
    :return: shifted position (x2, y2) and sum of weights within the window
    """
    x2 = None
    y2 = None
    sum_w = None
    '''
    ...
    your code ...
    ...
    '''
    return x2, y2, sum_w

def task_4_a():
    print("Task 4 (a) ...")
    img = cv.imread('../images/line.png')
    img_gray = None # convert the image into grayscale
    edges = None # detect the edges
    theta_res = None # set the resolution of theta
    d_res = None # set the distance resolution
    #_, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    '''
    ...
    your code ...
    ...
    '''

def task_4_b():
    print("Task 4 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''

##############################################
##############################################
##############################################

if __name__ == "__main__":
    #task_1_a()
    task_1_b() 
    task_3_a() 
    task_3_b() 
    task_3_c()
    