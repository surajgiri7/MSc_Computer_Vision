# importing all the required libraries
import cv2 as cv
import random as random
import numpy as np

img = 'bonn.png'

img_value = cv.imread(img)
gray_img = cv.cvtColor(img_value, cv.COLOR_BGR2GRAY)
cv.imshow('Original Image', gray_img)
cv.waitKey(0)
