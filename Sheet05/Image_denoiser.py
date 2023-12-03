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
    #warum 255 ? binary so betweem 0,255. 255 representing 1
    # This is a common approach in optimization problems where costs are used, so -np.log(rho)
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
            Denoised_I[y,x] = 1 - g.get_segment(nodes[index(img, x, y)[0]])

    return Denoised_I



def index_1(img, x, y):
    if x >= img.shape[1] or x < 0 or y >= img.shape[0] or y < 0:
        return None
    return (img.shape[1]*y+x, (x,y))

def unary_cost_1(img, p1, label, rho):
    return -np.log(rho) if img[p1[1],p1[0]] == label else -np.log(1 - rho)

def pairwise_cost_1(img, p1, p2):
    return 1 if img[p1[1],p1[0]] != img[p2[1],p2[0]] else 0

def grayscale_img_denoiser(img, rho):
    """
    :param img:
    :param rho:
    :return:
    """
    labels = np.unique(img)

    denoised_img = np.zeros_like(img)

    for label in labels:
        # 1) Define Graph
        size = img.shape[0] * img.shape[1]
        g = maxflow.Graph[float](size, size * 4)

        # 2) Add pixels as nodes
        nodes = g.add_nodes(size)

        # 3) Compute Unary cost and 4) Add terminal edges
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                i = index_1(img, x, y)
                g.add_tedge(nodes[i[0]], unary_cost_1(img, i[1], label, rho), 1-unary_cost_1(img, i[1], label, rho))

        # 5) Add Node edges
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                i = index_1(img, x, y)
                if x+1 < img.shape[1]:
                    ir = index_1(img, x+1, y)
                    g.add_edge(nodes[i[0]], nodes[ir[0]], pairwise_cost_1(img, i[1], ir[1]), pairwise_cost_1(img, i[1], ir[1]))
                if y+1 < img.shape[0]:
                    ib = index_1(img, x, y+1)
                    g.add_edge(nodes[i[0]], nodes[ib[0]], pairwise_cost_1(img, i[1], ib[1]), pairwise_cost_1(img, i[1], ib[1]))

        # 6) Maxflow
        g.maxflow()

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if g.get_segment(nodes[index_1(img, x, y)[0]]) == 0:
                    denoised_img[y,x] = label

    return denoised_img



def main():
    image_binary = cv2.imread('./images/noisy_binary_img.png', cv2.IMREAD_GRAYSCALE)
    image_grayscale = cv2.imread('./images/noisy_grayscale_img.png', cv2.IMREAD_GRAYSCALE)

    rho = 0.35
    pairwise_cost_same_values = np.arange(0.0, 1, step=0.1)
    pairwise_cost_diff_values = np.arange(0.0, 1, step=0.1)


    # Test binary_img_denoiser with different values
    for theta_s in pairwise_cost_same_values:
        for theta_d in pairwise_cost_diff_values:
            denoised_output = binary_img_denoiser(image_binary, rho, theta_s, theta_d)
            # Display or save original and denoised images
            cv2.imshow(f'Original Image', image_binary)
            cv2.imshow(f'Denoised Output (rho={rho}, theta_s={theta_s}, theta_d={theta_d})', denoised_output)
            cv2.waitKey(0)

    
    denoised_output = grayscale_img_denoiser(image_grayscale, rho)
    cv2.imshow(f'Original Image', image_grayscale)
    cv2.imshow(f'Denoised Output ', denoised_output)
    cv2.waitKey(0)
    

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
    
    """
        3.2.1 as the theta_s increases with the theta_d the noise is reduced but it is 
        really hard to see the image structure 
        3.2.2 till theta_s <= 0.2 with any theta_ds the image is very nosiy the image starts to better 
        and less noisy as theta_s = 0.3 and theta_d = 0.6, as we increase the theta_s from 0.3 until theta_s is 
        not 0.6 the image is very noisy. But as theta_s incrases the image structure is deformed. If one has to
        conclude, it can be said that the best combination is from theta_s >= 0.3 and theta_s <= 0.4 where 
        theta_d >= 0.6. Also the image gets very distored. 
    
    """


