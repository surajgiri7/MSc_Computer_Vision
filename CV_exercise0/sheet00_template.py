import cv2 as cv
import random as random
import numpy as np


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    img_value = cv.imread("bonn.png")
    cv.imshow(window_name, img_value)
    cv.waitKey(0)
    # cv.destroyAllWindows() 

def convert_into_intensity_image(window_name, img):
    img_value = cv.imread("bonn.png")
    img= cv.cvtColor(img_value, cv.COLOR_BGR2BGR )
    cv.imshow(window_name, img) 
    cv.waitKey(0)
    # cv.destroyAllWindows()

def random_patch(window_name, img):
    # Step 1: Read the original image
    image = cv.imread(img)

    # Step 2: Find the middle of the image
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2

    # Step 3: Create a 16x16 patch
    patch_size = 16
    top_left_x = center_x - (patch_size // 2)
    top_left_y = center_y - (patch_size // 2)
    bottom_right_x = top_left_x + patch_size
    bottom_right_y = top_left_y + patch_size

    # Step 4: Extract the 16x16 patch
    patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Step 5: Pick a random spot
    new_x = random.randint(0, width - patch_size)
    new_y = random.randint(0, height - patch_size)

    # Step 6: Place the patch in a random location
    image_with_patch = np.copy(image)
    image_with_patch[new_y:new_y + patch_size, new_x:new_x + patch_size] = patch

    # Step 7: Show the new picture
    cv.imshow('Image with Patch', image_with_patch)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rand_rect_ellipse(window_name, img):
    img = cv.imread(img)

    # Define the number of shapes you want to draw
    num_rectangles = 10
    num_ellipses = 10

    # Define colors for the shapes (in BGR format)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Draw 10 random rectangles
    for _ in range(num_rectangles):
        color = random.choice(colors)
        top_left = (random.randint(0, 500 - 10), random.randint(0, 300 - 10))
        bottom_right = (random.randint(top_left[0] + 10, 500), random.randint(top_left[1] + 10, 300))
        cv.rectangle(img, top_left, bottom_right, color, -1)  # -1 for filled rectangle

    # Draw 10 random ellipses
    for _ in range(num_ellipses):
        color = random.choice(colors)
        center = (random.randint(0, 500), random.randint(0, 300))
        axes = (random.randint(10, 100), random.randint(10, 100))
        angle = random.randint(0, 360)
        cv.ellipse(img, center, axes, angle, 0, 360, color, -1)  # -1 for filled ellipse

    # Display the img
    cv.imshow('Shapes', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':

    # set image path
    img = 'bonn.png'
    
    # 2a: read and display the image 

    # display_image('2 - a - Original Image', img)

    
    # 2b: display the intensity image
    # convert_into_intensity_image('2 - b - Intensity Image', img)

    # 2c: for loop to perform the operation
    # random_patch('2 - c - Reduced Intensity Image', img)
    """

    # 2d: one-line statement to perfom the operation above
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)    

    # 2e: Extract the center patch and place randomly in the image
    display_image('2 - e - Center Patch', img_patch)  
    
    # Random location of the patch for placement
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)  """


    # 2f: Draw random rectangles and ellipses
    rand_rect_ellipse('2 - f - Rectangles and Ellipses', img)
    """
    # destroy all windows 
    """
    cv.destroyAllWindows()
