#!/usr/bin/python3.5

import numpy as np
import cv2 as cv

'''
    load the image and foreground/background parts
    image: the original image
    background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''

def read_image(filename):
    image = cv.imread(filename) / 255.0
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[90:350, 110:250, :] = 1
    bb_width, bb_height = 140, 260
    background = image[bounding_box == 0].reshape((height * width - bb_width * bb_height, 3))
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))
   
    return image, foreground, background


class GMM(object): 


    def __init__(self, image, foreground, background):
        self.image = image
        self.foreground = foreground
        self.background = background
        self.data = None
        self.mu = None
        self.sigma = None 
        self.cov = None
        self.weight = np.ones(1)


    def gaussian_scores(self, data):
        
        pass


    def fit_single_gaussian(self, data): 
        mu = np.mean(data,axis=0) 
        cov = np.diag(np.var(data, axis=0)) 
        return mu, cov



    def estep(self, data):
        # TODO
        pass


    def mstep(self, data, r):
        # TODO
        pass

    def em_algorithm(self, data, n_iterations = 10):
        # TODO
        pass


    def split(self, epsilon=0.1):
        
        pi_k = []
        mu_list = []
        cov_mat = []
        sigma_list = []

        for x in range(len(self.mu)):
            # Generate two new means
            mu_1 = self.mu[x] + epsilon * self.sigma[x]
            mu_2 = self.mu[x] - epsilon * self.sigma[x]

            # Append the new means to the list
            mu_list.append(mu_1)
            mu_list.append(mu_2)

            # Duplicate the weight and divide by 2
            pi_k.append(self.weight[x] / 2)
            pi_k.append(self.weight[x] / 2)

            # Duplicate the diagonal covariance matrix
            cov_mat.append(np.diag(self.cov[x]))
            cov_mat.append(np.diag(self.cov[x]))

            # Append the original standard deviation to the list
            sigma_list.append(self.sigma[x])
            sigma_list.append(self.sigma[x])

        # Update GMM parameters
        self.mu = np.array(mu_list)
        self.sigma = np.array(sigma_list)
        self.weight = np.array(pi_k)
        self.cov = np.array(cov_mat)

            

    def probability(self, data):
        # TODO
        pass


    def sample(self):
        # TODO
        pass


    def train(self, data, n_splits):
        # TODO
        pass

if __name__ == '__main__':

    image, foreground, background = read_image('data/person.jpg')

    '''
    TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
    Hint: Slide 64
    '''
    gmm_background = GMM()