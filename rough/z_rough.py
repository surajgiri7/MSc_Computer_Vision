# import numpy as np
# import matplotlib.pyplot as plt 
# import utils 

# # Load the aligned landmark data 
# hands_aligned_train = 'data/hands_aligned_train.txt.new'

# data = utils.load_data(hands_aligned_train)
# shapes = utils.convert_samples_to_xy(data['samples'])

# # Reshape the data for PPCA
# N, D = shapes.shape[0], shapes.shape[1] * shapes.shape[2]
# X = shapes.reshape(N, D)

# # Function for PPCA
# def ppca(X, q):
#     N, D = X.shape
#     mu = np.mean(X, axis=0)
#     X_centered = X - mu
#     C = np.dot(X_centered.T, X_centered) / N

#     # Perform eigendecomposition
#     eigenvalues, eigenvectors = np.linalg.eigh(C)

#     # Choose the top q eigenvectors
#     W = eigenvectors[:, -q:]

#     # Estimate the noise variance
#     sigma_sq = np.sum(eigenvalues[:-q]) / (D - q)

#     return W, mu, sigma_sq

# # Find the number of components to preserve 90% of the energy
# total_energy = np.sum(np.linalg.eigvals(np.cov(X.T)))
# cumulative_energy = 0
# num_components = 0

# for eigenvalue in sorted(np.linalg.eigvals(np.cov(X.T)), reverse=True):
#     cumulative_energy += eigenvalue
#     num_components += 1
#     if cumulative_energy / total_energy >= 0.9:
#         break

# # Apply PPCA
# W, mu, sigma_sq = ppca(X, num_components)

# # Visualize mean shape
# mean_shape = mu.reshape(shapes.shape[1], shapes.shape[2])
# plt.scatter(mean_shape[:, 0], mean_shape[:, 1], label='Mean Shape')
# plt.title('Mean Shape')
# plt.legend()
# plt.show()

# # Visualize the effect of varying positive and negative weights for each principal component
# for i in range(num_components):
#     weights = np.linspace(-3, 3, 7)  # Vary weights from -3 to 3
#     plt.figure(figsize=(10, 6))
    
#     for w in weights:
#         shape_variation = mu + w * np.sqrt(sigma_sq) * W[:, i]
#         shape_variation = shape_variation.reshape(shapes.shape[1], shapes.shape[2])
#         plt.scatter(shape_variation[:, 0], shape_variation[:, 1], label=f'Weight = {w:.2f}')

#     plt.title(f'Effect of Varying Weights for Principal Component {i+1}')
#     plt.legend()
#     plt.show()


import numpy as np
import utils
import matplotlib.pyplot as plt
import utils

# ======================= PPCA =======================
def ppca(covariance, preservation_ratio=0.9):
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    total_energy = np.sum(eigvals)
    cumulative_energy = 0
    num_components = 0

    for eigenvalue in eigvals:
        cumulative_energy += eigenvalue
        num_components += 1
        if cumulative_energy / total_energy >= preservation_ratio:
            break

    principal_components = eigvecs[:, :num_components]
    sigma_sq = np.sum(eigvals[num_components:]) / (len(eigvals) - num_components)

    return principal_components, sigma_sq


# ======================= Covariance =======================
def create_covariance_matrix(kpts, mean_shape):
    centered_shapes = kpts - mean_shape
    covariance_matrix = np.cov(centered_shapes.T)
    return covariance_matrix


# ======================= Visualization =======================
def visualize_impact_of_pcs(mean, pcs, pc_weights):
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
    
    utils.visualize_hands(utils.convert_samples_to_xy(A[0]), "Difference between each positive weighted PCs ", delay=.4)

    B = mean + np.expand_dims(pcs * np.expand_dims(negative_K_wegihts, axis=1), axis=0)
    utils.visualize_hands(utils.convert_samples_to_xy(B[0]), "Difference between each negative weighted PCs", delay=.4)


# ======================= Training =======================
def train_statistical_shape_model(kpts):
    mean_shape = np.mean(kpts, axis=0)
    covariance_matrix = create_covariance_matrix(kpts, mean_shape)
    principal_components, sigma_sq = ppca(covariance_matrix)

    return mean_shape, principal_components, sigma_sq


# ======================= Reconstruct =======================
def reconstruct_test_shape(mean, pcs, pc_weights):
    reconstructed_shape = mean + pcs @ np.diag(pc_weights)
    return reconstructed_shape


# Example usage:
if __name__ == "__main__":
    # Load the aligned landmark data
    hands_aligned_train = 'data/hands_aligned_train.txt.new'

    data = utils.load_data(hands_aligned_train)
    shapes = utils.convert_samples_to_xy(data['samples'])
    
    # Reshape the data for PPCA
    N, D = shapes.shape[0], shapes.shape[1] * shapes.shape[2]
    X = shapes.reshape(N, D)

    # Train the statistical shape model
    mean_shape, principal_components, sigma_sq = train_statistical_shape_model(X)

    # Visualize the mean shape
    plt.scatter(mean_shape[0::2], mean_shape[1::2], label='Mean Shape', marker='o', color='black')
    plt.title('Mean Shape')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.legend()
    plt.show()

    # Visualize the impact of principal components
    pc_weights = np.zeros(principal_components.shape[1])  # Set weights to zeros for demonstration
    visualize_impact_of_pcs(mean_shape, principal_components, pc_weights)
