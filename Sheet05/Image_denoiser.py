import cv2
import numpy as np

# need to install PyMaxflow (https://pmneila.github.io/PyMaxflow)
import maxflow


def binary_img_denoiser(img, rho, pairwise_cost_same, pairwise_cost_diff):
    """

    :param img:
    :param rho:
    :param pairwise_cost_same:
    :param pairwise_cost_diff:
    :return:
    """
    # 1) Define Graph
    g = maxflow.Graph[float]()

    # 2) Add pixels as nodes

    # 3) Compute Unary cost

    # 4) Add terminal edges

    # 5) Add Node edges
    # Vertical Edges

    # Horizontal edges
    # (Keep in mind the structure of neighbourhood and set the weights according to the pairwise potential)

    # 6) Maxflow
    g.maxflow()


def grayscale_img_denoiser(img, rho):
    """

    :param img:
    :param rho:
    :return:
    """
    labels = np.unique(img)

    denoised_img = np.zeros_like(img)
    # Use Alpha expansion binary image for each label

    # 1) Define Graph

    # 2) Add pixels as nodes

    # 3) Compute Unary cost

    # 4) Add terminal edges

    # 5) Add Node edges

    # 6) Maxflow


def main():
    image_binary = cv2.imread('./images/noisy_binary_img.png', cv2.IMREAD_GRAYSCALE)
    image_grayscale = cv2.imread('./images/noisy_grayscale_img.png', cv2.IMREAD_GRAYSCALE)

    """
    Some notes on question 3
    - You can test binary_img_denoiser with the following range
            - pairwise_cost_same: np.arange(0.0, 1, step=0.1)
            - pairwise_cost_diff: np.arange(0.0, 1, step=0.1)
            - you can set rho to 0.35
        for better results you need more fine-grained step (such as 0.05).
    - Display the denoised output images for different values and
        based on these results report what is the impact of each parameter. 
    """


if __name__ == "__main__":
    main()


