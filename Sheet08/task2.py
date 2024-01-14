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
    print("Covariance Shape: \n", covariance_matrix.shape)
    print


    # Perform SVD
    U, s, Vt = np.linalg.svd(covariance_matrix.T @ covariance_matrix)
    print("L^2")
    print(s)

    # Select the minimum number of principal components preserving 90% of the energy
    n_components = np.where(np.cumsum(s / np.sum(s)) >= preservation_ratio)[0][0] + 1 # +1 because of zero indexing
    print("n_components: \n", n_components)

    # Calculate the principal components
    pcs = U[:, :n_components]
    print("pcs matrix") 
    print(pcs)
    # Calculate the variance
    D = covariance_matrix.shape[1]
    sigma_hat_square = np.sum(s[n_components:]) / (D - n_components)

    # Calculate the weights
    pc_weights = pcs @ np.sqrt(s[:n_components] - sigma_hat_square * np.eye(n_components))

    return pcs,sigma_hat_square, pc_weights




# ======================= Covariance =======================

def create_covariance_matrix(kpts, mean_shape):
    # ToDO
    W = kpts - mean_shape
    return W





# ======================= Visualization =======================

def visualize_impact_of_pcs(mean, pcs, pc_weights):
    # your part here
    utils.visualize_hands(utils.convert_samples_to_xy(np.expand_dims(mean, axis=0)), "mean", delay=2)
    print("after")
    # get positive and negative weights 
    v = 1
    positive_K_weights = pc_weights * v
    negative_K_wegihts = pc_weights * -v
    print("positive_K_weights")
    print(positive_K_weights)
    print("negative_K_wegihts")
    print(negative_K_wegihts)

    A = mean + np.sum(pcs * np.expand_dims(positive_K_weights, axis=1), axis=2)
    utils.visualize_hands(utils.convert_samples_to_xy(A), "Mean with Sum of positive weighted PCs", delay=0.1)

    B = mean + np.sum(pcs * np.expand_dims(negative_K_wegihts, axis=1), axis=2)
    utils.visualize_hands(utils.convert_samples_to_xy(B), "Mean with Sum of negative weighted PCs", delay=0.1)







# ======================= Training =======================
def train_statistical_shape_model(kpts):
    # Your code here 
    print("kpts shape") 
    print(kpts.shape)
    mean_shape = np.mean(kpts, axis=0)
    # print("Mean Shape: ", mean_shape)
    # print("Mean Shape Shape: ", mean_shape.shape)
    covariance = create_covariance_matrix(kpts, mean_shape)
    # print("Covariance: ", covariance)
    # print("Covariance Shape: ", covariance.shape)
    pcs,sigma_sq, phi = ppca(kpts)
    # print("Mean: \n", mean)
    # print("pcs: \n", pcs)
    # print("pc_weights: \n", pc_weights) 
    print("sigma_sq") 
    print(sigma_sq) 
    print("phi") 
    print(phi.shape) 
    visualize_impact_of_pcs(mean_shape,pcs, phi)  
    return mean_shape, sigma_sq,phi
    




# ======================= Reconstruct =======================
def reconstruct_test_shape(kpts, mean, pcs, pc_weight):
    print("Hik: ", pc_weight)

    weighted_pc = pcs * np.expand_dims(pc_weight, axis=1)
    weighted_pc = np.sum(weighted_pc, axis=2)  # Sum along the last axis to get shape (112, 1)
    # Now, weighted_pc has shape (112, 1)
    X = kpts + weighted_pc.T + mean
    utils.visualize_hands(utils.convert_samples_to_xy(X), "reconstruction")


