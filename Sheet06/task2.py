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

# https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function
def pdf(x, mu, sigma):
    # Calculate the numerator of the PDF formula
    diff = x - mu.reshape(1, mu.shape[0])
    scaled_diff = diff ** 2 / sigma.reshape(1, sigma.shape[0])
    numerator = np.exp(-0.5 * np.sum(scaled_diff, axis=1))

    # Calculate the denominator of the PDF formula
    denominator = np.sqrt((2 * np.pi) ** mu.shape[0] * np.prod(sigma))

    # Return the PDF
    return numerator / denominator

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

    
    


    def fit_single_gaussian(self, data): 
        mu = np.mean(data,axis=0) 
        cov = np.diag(np.var(data, axis=0)) 
        return mu, cov



    def estep(self):
        r = np.zeros((self.data.shape[0], self.mu.shape[0]))
        pdf_values = np.zeros((self.data.shape[0], self.mu.shape[0]))

        # Calculate PDF values and denominator
        for k in range(self.mu.shape[0]):
            pdf_values[:, k] += self.weight[k] * pdf(self.data, self.mu[k], self.sigma[k])
        denominator = np.sum(pdf_values, axis=1)

        # Calculate responsibilities
        for k in range(self.mu.shape[0]):
            r[:, k] = pdf_values[:, k] / denominator

        return r

    def mstep(self, r):
        for k in range(self.mu.shape[0]):
            r_sum = r[:, k].sum()
            
            # Update weight
            self.weight[k] = r_sum / r.sum()
            
            # Update mean
            weighted_data = r[:, k][:, np.newaxis] * self.data
            self.mu[k] = np.sum(weighted_data, axis=0) / r_sum
            
            # Update covariance matrix (sigma)
            diff = self.data - self.mu[k]
            squared_diff = np.power(diff, 2)
            self.sigma[k] = np.sum(r[:, k] * np.sum(squared_diff, axis=1)) / r_sum

    def em_algorithm(self, tau=0.8):
        # Initialize GMM parameters using fit_single_gaussian
        mu, sigma = self.fit_single_gaussian(self.background)
        self.mu = np.array([mu])
        self.sigma = np.array([sigma])

        # Train the GMM on the given data
        self.train(self.background)

        prob = self.probability(self.data)

        indices = np.where(prob > tau)

        # Manipulate the image based on the probabilities
        data *= self.image
        data = cv.cvtColor(data.astype(np.uint8), cv.COLOR_HSV2BGR)
        data[indices] = [0, 0, 0]
        self.image = data




    def split(self, epsilon=0.1):
        new_mu = []
        new_sigma = []
        new_weight = []

        for i in range(self.mu.shape[0]):
            mu_i = self.mu[i]
            sigma_i = self.sigma[i]
            weight_i = self.weight[i]

            mu_1, mu_2 = mu_i + sigma_i * epsilon, mu_i - sigma_i * epsilon
            sigma_1, sigma_2 = sigma_i, sigma_i
            weight_2, weight_1 = weight_i / 2, weight_i / 2

            new_mu.extend([mu_1, mu_2])
            new_sigma.extend([sigma_1, sigma_2])
            new_weight.extend([weight_1, weight_2])

        self.mu = np.array(new_mu).reshape(-1, 3)
        self.sigma = np.array(new_sigma).reshape(-1, 3)
        self.weight = np.array(new_weight)


            

    def probability(self, data):
        probabilities = np.zeros(data.shape[0] * data.shape[1])
        for k in range(self.mu.shape[0]):
            probabilities += self.weight[k] * pdf(data.reshape(data.shape[0] * data.shape[1], 3), self.mu[k], self.sigma[k])
        probabilities = probabilities.reshape(data.shape[0], data.shape[1])
        return probabilities / probabilities.max()


    def sample(self):
        # TODO
        pass


    def train(self, data, n_iterations=10, n_splits=3):
        self.data = data

        for i in range(n_iterations):
            if i < n_splits:
                self.split()

            responsibilities = self.estep()
            self.mstep(responsibilities)


if __name__ == '__main__':

    image, foreground, background = read_image('data/person.jpg')

    '''
    TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
    Hint: Slide 64
    '''
gmm_background = GMM(image, foreground, background)
gmm_background.em_algorithm(image)