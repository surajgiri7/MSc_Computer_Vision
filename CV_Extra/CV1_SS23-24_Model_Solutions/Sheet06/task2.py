#!/usr/bin/python3.5

import numpy as np
import cv2 as cv

'''
    load the face image and foreground/background parts
    image: the original image
    foreground/background: numpy arrays of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''


def read_image(filename):
    image = cv.imread(filename) / 255
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[90:350, 110:280, :] = 1
    bb_width, bb_height = 170, 260
    
    test = image * bounding_box 
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))
    background = image[bounding_box == 0].reshape((height * width - bb_width * bb_height, 3))
    return image, foreground, background

class GMM(object):

    def gaussian_scores(self, data):

        # work in log space for numerical stability
        scores = -0.5 * (data[None, :, :] - self.means[:, None, :]) ** 2 / self.variances[:, None, :]
        scores = scores - 0.5 * np.log(2 * np.pi * self.variances[:, None, :])
        scores = np.sum(scores,
                        axis=2)  # taking the sum of logarithms is much more stable than multiplying probabilities!

        return scores

    def fit_single_gaussian(self, data):
        self.weights = np.ones((1,), dtype=np.float64)  # dummy weight, Gaussian is treated as GMM with one component
        self.means = np.expand_dims(np.mean(data, axis=0), axis=0)

        # we have diagonal covariance matrices, so it's only the variance an we can store it as a vector with the same size as the mean
        self.variances = np.expand_dims(np.var(data, axis=0),
                                        axis=0) + 1e-6  # add a small epsilon to avoid zero variance

    def estep(self, data):
        scores = self.gaussian_scores(data) + np.log(self.weights)[:, None]  # this is again in log space for stability

        max_score = np.max(scores)
        r = np.exp(
            scores - max_score)  # another trick for stability: subtract the maximum. Note that in the original equation for r, this corresponds to omitting a constant in both, numerator and denominator, so the constant would anyway cancel. The exp function can now be computed with values much closer around 0 (more stable, less rounding errors)
        norm = np.sum(r, axis=0) + 1e-6
        r = r / norm[None, :] + 1e-6  # again add a small epsilon to avoid zeros
        return r

    def mstep(self, data, r):
        self.weights = np.sum(r, axis=1) / np.sum(r)
        self.means = np.sum(r[:, :, None] * data[None, :, :], axis=1) / np.sum(r, axis=1)[:, None]
        self.variances = np.sum(r[:, :, None] * (data[None, :, :] - self.means[:, None, :]) ** 2, axis=1) / np.sum(r,
                                                                                                                   axis=1)[
                                                                                                            :,
                                                                                                            None] + 1e-6  # avoid zero variance

    def em_algorithm(self, data, n_iterations=10):
        for i in range(n_iterations):
            r = self.estep(data)
            self.mstep(data, r)

    def split(self, epsilon=0.1):
        self.weights = np.concatenate((self.weights, self.weights), axis=0) / 2.0
        self.means = np.concatenate(
            (self.means - epsilon * self.variances ** 0.5, self.means + epsilon * self.variances ** 0.5), axis=0)
        self.variances = np.concatenate((self.variances, self.variances), axis=0)

    def probability(self, data):
        scores = self.gaussian_scores(data) + np.log(self.weights[:, None])

        prob = np.sum(np.exp(scores), axis=0)

        return prob

    def train(self, data, n_splits):
        self.fit_single_gaussian(data)
        for _ in range(n_splits):
            self.split()
            self.em_algorithm(data)


def main():
    image, foreground, background = read_image('data/person.jpg')

    '''
    TODO: compute p(x|w=background) for each image pixel and manipulate image such that everything below the threshold is black, display the resulting image
    Hint: Slide 64
    '''
    # gmm_foreground, gmm_background = GMM(), GMM()
    gmm_background = GMM()

    gmm_background.train(background, 3)
    data = image.reshape((image.shape[0] * image.shape[1], 3))
    prob_background = gmm_background.probability(data)
    img = data[:, :]

    img[prob_background > 5, :] = 0
    cv.imwrite('background_subtraction.png', img.reshape(image.shape[0], image.shape[1], 3) * 255)

if __name__ == '__main__':
    main()