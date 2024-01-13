import numpy as np
import utils

# ======================= PPCA =======================
def ppca(covariance, preservation_ratio=0.9):

    # Happy Coding! :)

    mean_shape = np.mean(covariance, axis=0)
    print("Mean Shape: \n", mean_shape)

    # Calculate the covariance matrix
    covariance_matrix = create_covariance_matrix(covariance, mean_shape)

    print("Covariance: \n", covariance_matrix)


    # Perform SVD
    U, s, Vt = np.linalg.svd(covariance_matrix.T @ covariance_matrix)

    # Select the minimum number of principal components preserving 90% of the energy
    n_components = np.where(np.cumsum(s / np.sum(s)) >= preservation_ratio)[0][0] + 1 # +1 because of zero indexing
    print("n_components: \n", n_components)

    # Calculate the principal components
    pcs = U[:, :n_components]

    # Calculate the variance
    D = covariance_matrix.shape[1]
    sigma_hat_square = np.sum(s[n_components:]**2) / (D - n_components)

    # Calculate the weights
    pc_weights = pcs * np.sqrt(s[:n_components]**2 - sigma_hat_square)

    return mean_shape, pcs, pc_weights




# ======================= Covariance =======================

def create_covariance_matrix(kpts, mean_shape):
    # ToDO
    W = kpts - mean_shape
    covariance = np.cov(W.T)
    return W
    pass





# ======================= Visualization =======================

def visualize_impact_of_pcs(mean, pcs, pc_weights):
    # your part here
    utils.visualize_hands(utils.convert_samples_to_xy(np.expand_dims(mean, axis=0)), "mean", delay=1)

    # get positive and negative weights
    v = np.sqrt(5)
    positive_K_weights = pc_weights * v
    negative_K_wegihts = pc_weights * -v

    A = mean + np.expand_dims(np.sum(pcs * np.expand_dims(positive_K_weights, axis=1), axis=0), axis=0)
    utils.visualize_hands(utils.convert_samples_to_xy(A), "Mean with Sum of positive weighted PCs", delay=1)

    B = mean + np.expand_dims(np.sum(pcs * np.expand_dims(negative_K_wegihts, axis=1), axis=0), axis=0)
    utils.visualize_hands(utils.convert_samples_to_xy(B), "Mean with Sum of negative weighted PCs", delay=1)

    A = mean + np.expand_dims(pcs * np.expand_dims(positive_K_weights, axis=1), axis=0)
    utils.visualize_hands(utils.convert_samples_to_xy(A[0]), "Difference between each positive weighted PCs", delay=.4)

    B = mean + np.expand_dims(pcs * np.expand_dims(negative_K_wegihts, axis=1), axis=0)
    utils.visualize_hands(utils.convert_samples_to_xy(B[0]), "Difference between each negative weighted PCs", delay=.4)


    pass





# ======================= Training =======================
def train_statistical_shape_model(kpts):
    # Your code here
    mean_shape = np.mean(kpts, axis=0)
    # print("Mean Shape: ", mean_shape)
    # print("Mean Shape Shape: ", mean_shape.shape)
    covariance = create_covariance_matrix(kpts, mean_shape)
    # print("Covariance: ", covariance)
    # print("Covariance Shape: ", covariance.shape)
    mean, pcs, pc_weights = ppca(kpts)
    # print("Mean: \n", mean)
    # print("pcs: \n", pcs)
    # print("pc_weights: \n", pc_weights)
    visualize_impact_of_pcs(mean, pcs, pc_weights)
    return mean, pcs, pc_weights
    pass




# ======================= Reconstruct =======================
def reconstruct_test_shape(kpts, mean, pcs, pc_weight):
    #ToDo
    pass
