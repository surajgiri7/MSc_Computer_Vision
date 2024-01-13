import numpy as np
import utils

# ======================= PPCA =======================
def ppca(covariance, preservation_ratio=0.9):

    # Happy Coding! :)

    # Perform SVD
    U, s, Vt = np.linalg.svd(covariance)

    # Select the minimum number of principal components preserving 90% of the energy
    n_components = np.where(np.cumsum(s / np.sum(s)) >= preservation_ratio)[0][0]

    # Extract the basis functions
    basis_functions = U[:, :n_components]

    # Calculate the mean shape
    mean_shape = np.mean(covariance, axis=1)
    print("Mean Shape: \n", mean_shape)

    # Calculate the sigma^2
    sigma_hat_square = np.sum((s[n_components:]**2)) / (covariance.shape[0] - n_components)

    # Calculate the phi_hat
    lambda_k = np.diag(np.sqrt(s[:n_components]**2))
    phi_hat = basis_functions @ (np.sqrt(lambda_k - sigma_hat_square * np.eye(n_components)))

    return phi_hat, mean_shape, sigma_hat_square




# ======================= Covariance =======================

def create_covariance_matrix(kpts, mean_shape):
    # ToDO
    W = kpts - mean_shape
    covariance = np.cov(W.T)
    return covariance
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
    mean, pcs, pc_weights = ppca(covariance)
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
