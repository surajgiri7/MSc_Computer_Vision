import numpy as np
import cv2
import random

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_Homography_RANSAC(good_matches, kp_1, kp_2, image_1, image_2, nSamples=4, nIterations=20, thresh=0.1):

    # --- YOUR CODE HERE ---#
    # TODO compute best H transformation using RANSAC algorithm

    #  /// RANSAC loop
    best_mse = 1e20
    for i in range(nIterations):

        print('iteration ' + str(i))

        # randomly select some keypoints
        rand_matches = [match[0] for match in random.sample(good_matches, nSamples)]

        pts1 = [[kp_1[match.queryIdx].pt[0], kp_1[match.queryIdx].pt[1]] for match in rand_matches]
        pts2 = [[kp_2[match.trainIdx].pt[0], kp_2[match.trainIdx].pt[1]] for match in rand_matches]

        hom = cv2.getPerspectiveTransform(np.float32(pts2), np.float32(pts1))
        warpedImg2 = cv2.warpPerspective(image_2, hom, (image_1.shape[1], image_1.shape[0]))

        total_mse = 0
        inliers_count = 0

        for test_kp in [kp for kp in kp_1 if kp.size > 0]:

            size = test_kp.size

            min_r, max_r, min_c, max_c = test_kp.pt[1] - size / 2, test_kp.pt[1] + size / 2, test_kp.pt[0] - size / 2, \
                                         test_kp.pt[0] + size / 2

            patch1 = image_1[int(min_r):int(max_r), int(min_c):int(max_c), :].astype(np.float32)
            patch2 = warpedImg2[int(min_r):int(max_r), int(min_c):int(max_c), :].astype(np.float32)

            diff = patch1 - patch2

            mse = np.sum(np.multiply(diff, diff))

            mse /= (size * size * 3 * 255 * 255)

            if mse < thresh:
                inliers_count += 1
                total_mse += mse

        total_mse /= inliers_count + 1e-12

        if total_mse < best_mse:
            best_H = hom
            best_mse = total_mse

    return best_H


def get_best_match(des_1, des_2, thr=0.3):

    # --- YOUR CODE HERE ---#
    # TODO get best matches

    # own implementation of matching
    dist = np.expand_dims(des_1, axis=1) - np.expand_dims(des_2, axis=0)  # 1032, 1079, 128
    #print(dist.shape)
    dist = np.sum(np.multiply(dist, dist), axis=2)
    #print(dist.shape)  # 1032, 1079
    match12 = np.argmin(dist, axis=1)
    match21 = np.argmin(dist, axis=0)
    distances12 = np.min(dist, axis=1)
    # distances21 = np.min(dist, axis=0)
    #print(match12.shape)
    good_matches = []

    for queryIdx, trainIdx in enumerate(match12):
        if match21[trainIdx] == queryIdx and thr * dist[queryIdx, np.argpartition(dist[queryIdx, :], 2)[1]] > \
                distances12[queryIdx]:

            good_matches.append([cv2.DMatch(queryIdx, trainIdx, distances12[queryIdx])])

    return good_matches

def task_03():

    # --- YOUR CODE HERE ---#
    # TODO read the image and extract key-points

    image_1 = cv2.imread(r'./data/Mountain_1.png')
    image_2 = cv2.imread(r'./data/Mountain_2.png')

    sift = cv2.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(image_1, None)
    kp_2, des_2 = sift.detectAndCompute(image_2, None)
    #print(des_1.shape)  # 1032, 128
    #print(des_2.shape)  # 1079, 128

    # --- YOUR CODE HERE ---#
    # TODO get the best matching and visualize them
    good_matches = get_best_match(des_1, des_2, thr=0.3)

    img3 = cv2.drawMatchesKnn(image_1, kp_1, image_2, kp_2, good_matches, None, flags=2)
    # cv2.imwrite('matched_keypoints.png', img3)
    cv2.imshow('matched keypoints', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- YOUR CODE HERE ---#
    # TODO implement RANSAC algorithm to compute the best transformation H
    H = compute_Homography_RANSAC(good_matches, kp_1, kp_2, image_1, image_2)

    # --- YOUR CODE HERE ---#
    # TODO Stitch the images and visualize the result

    warpedImg2 = cv2.warpPerspective(image_2, H, (image_1.shape[1] + image_2.shape[1] + image_2.shape[1] // 4,
                                                  image_1.shape[0]))

    warpedImg2_mask = np.zeros_like(warpedImg2)
    warpedImg2_mask[0:image_1.shape[0], 0:image_1.shape[1]] = image_1

    mask = np.expand_dims((np.sum(warpedImg2, axis=2) == 0).astype(np.uint8), axis=2)
    warpedImg2 = warpedImg2 + np.multiply(warpedImg2_mask, mask)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', warpedImg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    task_03()

