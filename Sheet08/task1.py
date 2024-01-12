import numpy as np
import utils



# ========================== Mean =============================
def calculate_mean_shape(kpts):
    # ToDO
    # convert list of keypoints to numpy array for easier manipulation
    kpts_array = np.array(kpts)
    
    # calculate mean along the first axis (i.e., for each coordinate across all shapes)
    mean_shape = np.mean(kpts_array, axis=0)

    # converting the mean shape to the original format
    mean_shape = np.expand_dims(mean_shape, axis=0)
    
    return mean_shape
    # pass



# ====================== Main Step ===========================
def procrustres_analysis_step(kpts, reference_mean):
    # Happy Coding
    reference_mean = np.squeeze(reference_mean, axis=0)
    # print("|||||")
    # print(reference_mean)
    # print(reference_mean.shape)
    # print("|||||")
    for i, kpt in enumerate(kpts):
        # Centering the shapes
        centroid_kpt = np.mean(kpt, axis=0)
        centroid_mean = np.mean(reference_mean, axis=0)
        translated_kpt = kpt - centroid_kpt
        translated_mean = reference_mean - centroid_mean

        # Implementing SVD decomposition 
        covariance_matrix = np.dot(translated_kpt.T, translated_mean)
        U, D, V = np.linalg.svd(covariance_matrix)
        rotation_matrix = np.dot(V, U.T)

        # Calculating the translation vector
        translation_vector = -np.dot(rotation_matrix, centroid_kpt.T) + centroid_mean.T
        translation_vector = translation_vector.reshape(2, 1)

        # Applying the rotation and translation transformation to each shape
        kpts[i] = np.dot(rotation_matrix, kpt.T + translation_vector).T
    
    return kpts

    pass



# =========================== Error ====================================

def compute_avg_error(kpts, mean_shape):
    # ToDo
    diffs = kpts - mean_shape
    rms_error = np.sqrt(np.mean(diffs ** 2))
    return rms_error
    pass




# ============================ Procrustres ===============================

def procrustres_analysis(kpts, max_iter=int(100)):
    kpts_array = utils.convert_samples_to_xy(kpts)
    reference_mean = calculate_mean_shape(kpts_array)
    print("!!!!!" * 6)
    print("Reference Mean: ")
    print(reference_mean)
    print(reference_mean.shape)
    print("!!!!!" * 6)
    for i in range(max_iter):
        print(f'Iteration {i}')
        ##################### Your Part Here #####################
        # aligning to the mean shape
        aligned_kpts = procrustres_analysis_step(kpts_array, reference_mean)
        
        # calculating the new mean shape
        new_mean = calculate_mean_shape(aligned_kpts)
        # print("$$$$$")
        # print(new_mean)
        # print(new_mean.shape)
        # print("$$$$$")

        # calculating the error
        error = compute_avg_error(aligned_kpts, new_mean)
        print("RMS Error: ", error)
        print("$$$$$" * 6)
        ##########################################################

    # visualize
    utils.visualize_hands(aligned_kpts, "Shapes After Alignment", delay=0.1)  
    utils.visualize_hands(calculate_mean_shape(aligned_kpts), "Mean Shape After Alignment", delay=0.5)
    return aligned_kpts