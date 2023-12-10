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
    background = image[bounding_box == 0].reshape(
        (height * width - bb_width * bb_height, 3))
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))

    return image, foreground, background

# https://math.stackexchange.com/questions/434629/3-d-generalization-of-the-gaussian-point-spread-function
def pdf(x, mu, sigma):
    numerator = np.exp(-0.5 *
                       np.sum(
                           (x-mu.reshape(1, mu.shape[0])) ** 2 *
                           (1/(sigma.reshape(1, sigma.shape[0]))), axis=1)
                       )
    denominator = np.sqrt(((2*np.pi) ** 3) * np.prod(sigma))
    return numerator/denominator


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
        mu = np.mean(data, axis=0)
        cov = np.var(data, axis=0)
        return mu, cov

    def estep(self):
        r = np.zeros((self.data.shape[0], self.mu.shape[0]))
        pdf_values = np.zeros((self.data.shape[0], self.mu.shape[0]))

        # Calculate PDF values and denominator
        for k in range(self.mu.shape[0]):
            pdf_values[:, k] = self.weight[k] * pdf(self.data, self.mu[k], self.sigma[k])
        denominator = np.sum(pdf_values, axis=1)

        # Calculate responsibilities
        for k in range(self.mu.shape[0]):
            r[:, k] = pdf_values[:, k] / denominator

        return r

    def mstep(self, r):
        weight = r.sum()

        for k in range(self.mu.shape[0]):
            r_sum = r[:,k].sum()
            print("|||||")
            print(weight)
            self.weight[k] = r_sum / weight
            self.mu[k] = np.sum(r[:,k].reshape(r.shape[0], 1) * self.data, axis=0) / r_sum
            diff = self.data - self.mu[k]
            self.sigma[k] = np.sum(r[:, k] * (diff ** 2).sum(axis=1)) / r_sum

    def em_algorithm(self, tau=0.8):
        self.initialize_parameters(self.background)
        self.train(self.background)
        prob = self.probability(self.image)
        self.segment_image(prob, tau)
        self.display_image(self.image.astype(np.uint8))

    def initialize_parameters(self, data):
        mu, sigma = self.fit_single_gaussian(data)
        self.mu = np.array([mu])
        self.sigma = np.array([sigma])

    def probability(self, image):
        flattened_image = image.reshape(image.shape[0] * image.shape[1], 3)
        prob = np.zeros(flattened_image.shape[0])
        for k in range(self.mu.shape[0]):
            prob += self.weight[k] * pdf(flattened_image, self.mu[k], self.sigma[k])
        return prob.reshape(image.shape[0], image.shape[1]) / prob.max()

    def segment_image(self, prob, tau):
        indices = np.where(prob > tau)
        self.image *= 255
        self.image = cv.cvtColor(self.image.astype(np.uint8), cv.COLOR_HSV2BGR)
        self.image[indices] = [0,0,0]

    def display_image(self, img):
        cv.imshow('image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def split(self, epsilon = 0.1):
        mu = np.zeros((self.mu.shape[0] * 2, self.mu.shape[1]))
        sigma = np.zeros((self.sigma.shape[0] * 2, self.sigma.shape[1]))
        weight = np.zeros(self.weight.shape[0] * 2)

        for i in range(self.mu.shape[0]):
            mu_i, sigma_i, weight_i = self.mu[i], self.sigma[i], self.weight[i]
            mu[2*i], mu[2*i+1] = mu_i + sigma_i * epsilon, mu_i - sigma_i * epsilon
            sigma[2*i], sigma[2*i+1] = sigma_i, sigma_i
            weight[2*i], weight[2*i+1] = weight_i / 2, weight_i / 2

        self.mu, self.sigma, self.weight = mu, sigma, weight

    def sample(self):
        # TODO
        pass

    def train(self, data, n_iterations=10, k_splits=3):
        self.data = data
        for i in range(n_iterations):
            if i < k_splits:
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
    gmm_background.em_algorithm()