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



# def grayscale_img_denoiser(I,rho=0.6):
#     labels = np.unique(I).tolist()

#     Denoised_I = np.zeros_like(I)
#     Denoised_I = I.copy()
#     ### Use Alpha expansion binary image for each label
#     #I = I/I.max()
#     num_labels = len(labels)
#     D = np.abs(I.reshape(I.shape + (1,)) - np.array(labels).reshape((1, 1, -1)))
#     D = np.where(D == 0, rho, (1-rho)/2)
#     V = 255 * np.eye(3)

#     max_iter = 50
#     better_energy = np.inf
#     # Stop when energy is not changed or reached max iteration.
#     for i in range(max_iter):
#         improved = False
#         # Iterate through the labels.
#         for alpha in labels:
#             # Create graph and Caculate the energy.
#             energy, Denoised_I = calculate_energy(alpha, D, V, Denoised_I, rho)
#             # Check if the better energy has been improved.
#             if energy < better_energy:
#                 better_energy = energy
#                 improved = True
#         # Finish the minimization when energy is not decreased.
#         if not improved:
#             break

#     cv2.imshow('Original Img', I), \
#     cv2.imshow('Denoised Img', Denoised_I), cv2.waitKey(0), cv2.destroyAllWindows()

#     return

# def calculate_energy(alpha, D, V, Denoised_I, rho):
#     ### 1) Define Graph
#     g = maxflow.Graph[float]()

#     row, col = Denoised_I.shape
#     ### 2) Add pixels as nodes
#     nodeids = g.add_grid_nodes((row, col))
#     ### 3) Compute Unary cost
#     ### 4) Add terminal edges
#     #g.add_grid_tedges(nodeids, rho, (1-rho)/2)
#     label_dict = {0:0, 128:1, 255:2}

#     ### 5) Add Node edges
#     ### Vertical Edges
#     for c in range(col):
#         for r in range(row-1):
#             curr_state = Denoised_I[r, c]
#             if curr_state == alpha:
#                 # Add unary cost from label to source and sink
#                 g.add_tedge(nodeids[r, c], D[r,c,label_dict[alpha]], np.inf)
#                 if Denoised_I[r+1, c] != alpha: # for alpha and else
#                     dist_state_alpha = V[label_dict[Denoised_I[r+1, c]], label_dict[alpha]]
#                     g.add_edge(nodeids[r, c], nodeids[r+1, c], 0, dist_state_alpha)
#             else:
#                 g.add_tedge(nodeids[r, c], D[r,c,label_dict[alpha]], D[r,c,label_dict[curr_state]])
#                 if Denoised_I[r+1, c] == alpha: # for else and alpha
#                     dist_state_alpha = V[label_dict[Denoised_I[r, c]], label_dict[alpha]]
#                     g.add_edge(nodeids[r, c], nodeids[r+1, c], dist_state_alpha, 0)
#                 else: # for else and else
#                     if Denoised_I[r, c] == Denoised_I[r+1, c]:
#                         dist_state_alpha = V[label_dict[Denoised_I[r, c]], label_dict[Denoised_I[r+1, c]]]
#                         g.add_edge(nodeids[r, c], nodeids[r + 1, c], dist_state_alpha, dist_state_alpha)
#                     else:
#                         dist = V[label_dict[Denoised_I[r, c]], label_dict[Denoised_I[r+1, c]]]
#                         curr = V[label_dict[Denoised_I[r, c]], label_dict[alpha]]
#                         next = V[label_dict[Denoised_I[r+1, c]], label_dict[alpha]]
#                         # Add an extra node between two nodes that both are not alpha
#                         extra_node = g.add_nodes(1)
#                         g.add_tedge(extra_node, 0, dist)
#                         g.add_edge(nodeids[r, c], extra_node, curr, np.inf)
#                         g.add_edge(nodeids[r+1, c], extra_node, np.inf, next)
#     ### Horizontal edges
#     # (Keep in mind the stucture of neighbourhood and set the weights according to the pairwise potential)
#     for r in range(row):
#         for c in range(col - 1):
#             curr_state = Denoised_I[r, c]
#             if curr_state == alpha:
#                 if Denoised_I[r, c+1] != alpha:
#                     dist_state_alpha = V[label_dict[alpha], label_dict[Denoised_I[r, c+1]]]
#                     g.add_edge(nodeids[r, c], nodeids[r, c+1], 0, dist_state_alpha)
#             else:
#                 if Denoised_I[r, c+1] == alpha:
#                     dist_state_alpha = V[label_dict[Denoised_I[r, c]], label_dict[alpha]]
#                     g.add_edge(nodeids[r, c], nodeids[r, c+1], dist_state_alpha, 0)
#                 else:
#                     if Denoised_I[r, c] == Denoised_I[r, c+1]:
#                         dist_state_alpha = V[label_dict[Denoised_I[r, c]], label_dict[Denoised_I[r, c+1]]]
#                         g.add_edge(nodeids[r, c], nodeids[r, c+1], dist_state_alpha, dist_state_alpha)
#                     else:
#                         dist = V[label_dict[Denoised_I[r, c]], label_dict[Denoised_I[r, c+1]]]
#                         curr = V[label_dict[Denoised_I[r, c]], label_dict[alpha]]
#                         next = V[label_dict[Denoised_I[r, c+1]], label_dict[alpha]]
#                         extra_node = g.add_nodes(1)
#                         g.add_tedge(extra_node, 0, dist)
#                         g.add_edge(nodeids[r, c], extra_node, curr, np.inf)
#                         g.add_edge(nodeids[r, c+1], extra_node, np.inf, next)

#     ### 6) Maxflow
#     energy = g.maxflow()

#     segments = g.get_grid_segments(nodeids)
#     segments = np.logical_not(segments)
#     for i in range(row):
#         for j in range(col):
#             if segments[i,j]:
#                 Denoised_I[i, j] = alpha

#     return energy, Denoised_I



def main():
    image_binary = cv2.imread('./images/noisy_binary_img.png', cv2.IMREAD_GRAYSCALE)
    image_grayscale = cv2.imread('./images/noisy_grayscale_img.png', cv2.IMREAD_GRAYSCALE)

    rho = 0.35
    pairwise_cost_same_values = np.arange(0.0, 1, step=0.1)
    pairwise_cost_diff_values = np.arange(0.0, 1, step=0.1)


    # Test binary_img_denoiser with different values
    """
    for theta_s in pairwise_cost_same_values:
        for theta_d in pairwise_cost_diff_values:
            denoised_output = binary_img_denoiser(image_binary, rho, theta_s, theta_d)
            # Display or save original and denoised images
            cv2.imshow(f'Original Image', image_binary)
            cv2.imshow(f'Denoised Output (rho={rho}, theta_s={theta_s}, theta_d={theta_d})', denoised_output)
            cv2.waitKey(0)
    """
    # grayscale_img_denoiser(image_grayscale,rho=0.6)

    for theta_s in pairwise_cost_same_values:
        for theta_d in pairwise_cost_diff_values:
            denoised_output = grayscale_img_denoiser(image_grayscale, rho)
            # Display or save original and denoised images
            cv2.imshow(f'Original Image', image_binary)
            cv2.imshow(f'Denoised Output (rho={rho}, theta_s={theta_s}, theta_d={theta_d})', denoised_output)
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


