import cv2
import numpy as np

# need to install PyMaxflow (https://pmneila.github.io/PyMaxflow)
import maxflow


def index(img, x, y):
    if x >= img.shape[1] or x < 0 or y >= img.shape[0] or y < 0:
        return None
    return (img.shape[1]*y+x, (x,y))

def pairwise_cost(img, p1, p2, pairwise_cost_same, pairwise_cost_diff):
    return pairwise_cost_same if (img[p1[1],p1[0]] == img[p2[1],p2[0]]) else pairwise_cost_diff

def unary_cost(img, p1, rho):
    return -np.log(1-rho) if (img[p1[1],p1[0]] == 255) else -np.log(rho)

def add_edges(img, g, nodes, rho, pairwise_cost_same, pairwise_cost_diff):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            i, il, ir, it, ib = index(img, x, y), index(img, x-1, y), index(img, x+1, y), index(img, x, y-1), index(img, x, y+1)

            if ir is not None:
                g.add_edge(nodes[i[0]], nodes[ir[0]], pairwise_cost(img, i[1],ir[1], pairwise_cost_same, pairwise_cost_diff), pairwise_cost(img, i[1],ir[1], pairwise_cost_same, pairwise_cost_diff))

            if ib is not None:
                g.add_edge(nodes[i[0]],nodes[ib[0]], pairwise_cost(img, i[1],ib[1], pairwise_cost_same, pairwise_cost_diff), pairwise_cost(img, i[1],ib[1], pairwise_cost_same, pairwise_cost_diff))

            g.add_tedge(nodes[i[0]], 1-unary_cost(img, i[1], rho), unary_cost(img, i[1], rho))

def binary_img_denoiser(img, rho, pairwise_cost_same, pairwise_cost_diff):
    size = img.shape[0] * img.shape[1]
    g = maxflow.Graph[float](size, size * 4)
    nodes = g.add_nodes(size)

    add_edges(img, g, nodes, rho, pairwise_cost_same, pairwise_cost_diff)

    g.maxflow()

    Denoised_I = img.copy().astype(np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            Denoised_I[y,x] = g.get_segment(nodes[index(img, x, y)[0]])

    return Denoised_I



    




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

    rho = 0.35
    pairwise_cost_same_values = np.arange(0.0, 1, step=0.1)
    pairwise_cost_diff_values = np.arange(0.0, 1, step=0.1)

    # Test binary_img_denoiser with different values
    for pairwise_cost_same in pairwise_cost_same_values:
        for pairwise_cost_diff in pairwise_cost_diff_values:
            denoised_output = binary_img_denoiser(image_binary, rho, pairwise_cost_same, pairwise_cost_diff)
            # Display or save denoised images
            cv2.imshow(f'Denoised Output (rho={rho}, pairwise_cost_same={pairwise_cost_same}, pairwise_cost_diff={pairwise_cost_diff})', denoised_output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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


