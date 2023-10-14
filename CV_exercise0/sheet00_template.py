import cv2 as cv
import random as random
import numpy as np

# 1. Function to read the image using imread and display it
def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    img_value = cv.imread(img)
    cv.imshow(window_name, img_value)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 2. Function to convert the image into intensity image and display it
def convert_into_intensity_image(window_name, img):
    """
    Displays the imtensity image with given window name.
    :param window_name: name of the window
    :param img: image object to convert to intensity image and display
    """
    img_value = cv.imread(img)
    intensity_image = cv.cvtColor(img_value, cv.COLOR_BGR2GRAY)
    cv.imshow(window_name, intensity_image) 
    cv.waitKey(0)
    cv.destroyAllWindows()


# 3. Function to multiply the imtensity of the image by 0.5 and 
# subtract it from each color channel new (R, G, B) values are
# (max(R - 0.5I, 0), max(G - 0.5I, 0), max(B - 0.5I, 0)).
def multiply_intensity_and_subtract_RGB(window_name, img):
    img_value = cv.imread(img)
    intensity_image = cv.cvtColor(img_value, cv.COLOR_BGR2GRAY)
    height, width, _ = img_value.shape
    print(height, width)
    # convertign the image to three channel image
    three_channel_img = np.stack((intensity_image,)*3, axis=-1)
    for y in range(height):
        for x in range(width):
            for c in range(3):
                print(f"Original Intensity Value for pixel {y}, {x} : ",img_value[y, x, c])
                img_value[y, x, c] = max((img_value[y, x, c] - 0.5 * three_channel_img[y, x]), 0)
                img_value[y, x, c] = max((img_value[y, x, c] - 0.5), 0)
                print(f"New Intensity Value for {y}, {x}: ",three_channel_img[y, x, c])
    cv.imshow(window_name, img_value)
    cv.waitKey(0)
    cv.destroyAllWindows()




    # img_cpy = np.copy(intensity_image)
    # # three_channel_img = np.stack((img_cpy,)*3, axis=-1)
    # print(three_channel_img.shape)
    # for y in range(height):
    #     for x in range(width):
    #         for c in range(3):
    #             print(f"Original Intensity Value for pixel {y}, {x} : ",img_value[y, x, c])
    #             img_cpy[y, x, c] = max((img_value[y, x, c] - 0.5 * three_channel_img[y, x]), 0)
    #             print("New Intensity Value: ",three_channel_img[y, x, c])
    # cv.imshow(window_name, img_cpy)
    # cv.waitKey(0)
    # cv.destroyAllWindows() 
    
    
    

# 4. Function to perform the operation in 3. in one line
def one_line_multiply_intensity_and_subtract_RGB(window_name, img):
    img_value = cv.imread(img)


# 5. Function to extract a 16x16 patch from the center of the image
# and place it randomly in the image
def pick_random_patch(window_name, img):
    img = cv.imread(img)

    height, width, _ = img.shape
    center_x, center_y = width // 2, height // 2 # finding the center of the image

    patch_size = 16
    top_left_x = center_x - (patch_size // 2)
    top_left_y = center_y - (patch_size // 2)
    bottom_right_x = top_left_x + patch_size
    bottom_right_y = top_left_y + patch_size

    patch = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    new_x = random.randint(0, width - patch_size)
    new_y = random.randint(0, height - patch_size)

    img_with_patch = np.copy(img)
    img_with_patch[new_y:new_y + patch_size, new_x:new_x + patch_size] = patch

    cv.imshow(window_name, img_with_patch)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rand_rect_ellipse(window_name, img):
    img = cv.imread(img)

    # Define the number of shapes you want to draw
    num_rectangles = 10
    num_ellipses = 10

    # Define colors for the shapes (in BGR format)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

    # Drawing 10 random rectangles
    for _ in range(num_rectangles):
        color = random.choice(colors)
        top_left = (random.randint(0, 500 - 10), random.randint(0, 300 - 10))
        bottom_right = (random.randint(top_left[0] + 10, 500), random.randint(top_left[1] + 10, 300))
        cv.rectangle(img, top_left, bottom_right, color, -1)  # -1 for filled rectangle

    # Drawing 10 random ellipses
    for _ in range(num_ellipses):
        color = random.choice(colors)
        center = (random.randint(0, 500), random.randint(0, 300))
        axes = (random.randint(10, 100), random.randint(10, 100))
        angle = random.randint(0, 360)
        cv.ellipse(img, center, axes, angle, 0, 360, color, -1)  # -1 for filled ellipse

    # Display the img
    cv.imshow(window_name, img)
    cv.waitKey(0)

if __name__ == '__main__':

    # set image path
    img = 'bonn.png'
    
    # # 2a: read and display the image 
    # display_image('2 - a - Original Image', img)

    
    # 2b: display the intensity image
    convert_into_intensity_image('2 - b - Intensity Image', img)

    # 2c: for loop to perform the operation
    multiply_intensity_and_subtract_RGB('2 - c - Reduced Intensity Image', img)

    # 2d: one-line statement to perfom the operation above
    # display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)    

    # # 2e: Extract the center patch and place randomly in the image
    # pick_random_patch('2 - e - Center Patch', img)  

    # # 2f: Draw random rectangles and ellipses
    # rand_rect_ellipse('2 - f - Rectangles and Ellipses', img)

    # destroy all windows 
    cv.destroyAllWindows()
