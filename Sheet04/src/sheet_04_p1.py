import numpy as np
import cv2 as cv

# Task 4.1.1
# function to compute the second smallest eigenvector of the affinity matrix W


def second_smallest_eig_vec(W):
    """
    :param W: affinity matrix
    :return: second smallest eigenvector of W
    """
    # Computing the degree matrix D which is a diagonal matrix where
    # each diagonal element Dii is the sum of the
    # elements of the ith row of the affinity matrix W.
    D = np.diag(np.sum(W, axis=1))
    print("Degree matrix: \n", D)

    # calculating D^(-1/2)
    D_sqrt_inv = np.sqrt(np.linalg.inv(D))

    # Calculating the normalized affinity matrix W_norm = D^(-1/2) * W * D^(-1/2)
    W_norm = np.dot(np.dot(D_sqrt_inv, D - W), D_sqrt_inv)

    # Calculating the eigenvector y using cv2.eigen
    state, eig_val, eig_vec = cv.eigen(W_norm)

    # Getting the indices that would sort the eigenvalues in ascending order
    idx = np.argsort(eig_val, axis=0)

    # Get the second smallest eigenvalue and its corresponding eigenvector
    second_smallest_eig_val = eig_val[idx[1]]
    second_smallest_eig_vec = eig_vec[:, idx[1]]

    # Converting the eigenvector to the original form using z = D^(-1/2) * y
    eig_vec_y = np.dot(D_sqrt_inv, second_smallest_eig_vec)

    print("Second smallest eigenvalue: \n", second_smallest_eig_val)
    print("Second smallest eigenvector: \n", second_smallest_eig_vec)
    print("Second smallest eigenvector in original form: \n", eig_vec_y)

    return eig_vec_y


def compute_Ncut_and_clusters(W, eig_vec_y):
    # Compute the clusters
    C1 = [i for i in range(len(eig_vec_y)) if eig_vec_y[i] > 0]
    C2 = [i for i in range(len(eig_vec_y)) if eig_vec_y[i] <= 0]

    # Computing the normalized cut
    # sum of all elements in the cut
    cut = np.sum([W[i, j] for i in C1 for j in C2])
    # sum of all elements in C1
    vol_C1 = np.sum([W[i, j] for i in C1 for j in range(W.shape[1])])
    # sum of all elements in C2
    vol_C2 = np.sum([W[i, j] for i in C2 for j in range(W.shape[1])])

    # Computing the normalized cut
    ncut = cut / vol_C1 + cut / vol_C2

    return C1, C2, ncut


if __name__ == "__main__":
    # At first creating the affinity matrix W based on the given graph
    W = np.array([
        [0.0, 1.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.2, 0.1, 0.0, 1.0, 0.0, 1.0, 0.3, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.3, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    ])

    # Computing the second smallest eigenvector of W
    second_smallest_eig_vec(W)

    # Computing the Ncut value and clusters
    eig_vec_y = second_smallest_eig_vec(W)
    C1, C2, ncut = compute_Ncut_and_clusters(W, eig_vec_y)

    # creating the mapping from from numbers to original labels
    mapping = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}

    # Printing the clusters
    print("Cluster 1: ", [mapping[i] for i in C1])
    print("Cluster 2: ", [mapping[i] for i in C2])
    print("Ncut: ", ncut)
