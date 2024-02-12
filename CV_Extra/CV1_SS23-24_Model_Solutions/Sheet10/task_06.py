import matplotlib.pyplot as plt
import numpy as np
import cv2

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def drawEpipolar(im1, im2, corr1, corr2, fundMat):

    # --- YOUR CODE HERE ---#
    # TODO visualize the corresponding points and the draw the Epipolar lines

    _, w_1, _ = im1.shape
    _, w_2, _ = im2.shape

    for i in range(corr1.shape[0]):

        color = (np.random.randint(1, 255), np.random.randint(1, 255), np.random.randint(1, 255))

        cv2.circle(im2, (int(corr2[i, 0]), int(corr2[i, 1])), 30, color, thickness=8, lineType=8, shift=0)
        cv2.drawMarker(im2, (int(corr2[i, 0]), int(corr2[i, 1])), color, cv2.MARKER_CROSS, 15, 4)

        left_points = np.array([corr1[i, 0], corr1[i, 1], 1]).reshape(3, 1)
        right_points = np.dot(fundMat, left_points)[:, 0]  # ax+by+c=0
        x0, y0 = map(int, [0, -right_points[2] / right_points[1]])
        x1, y1 = map(int, [w_2, -(right_points[2] + right_points[0] * w_2) / right_points[1]])

        # You can also use the function cv2.computeCorrespondEpilines()
        cv2.line(im2, (x0, y0), (x1, y1), color, 4)

        cv2.circle(im1, (int(corr1[i, 0]), int(corr1[i, 1])), 30, color, thickness=8, shift=0)
        cv2.drawMarker(im1, (int(corr1[i, 0]), int(corr1[i, 1])), color, cv2.MARKER_CROSS, 15, 4)

        right_points = np.array([corr2[i, 0], corr2[i, 1], 1]).reshape(3, 1)
        left_points = np.dot(fundMat.T, right_points)[:, 0]

        x0, y0 = map(int, [0, -left_points[2] / left_points[1]])
        x1, y1 = map(int, [w_1, -(left_points[2] + left_points[0] * w_1) / left_points[1]])
        cv2.line(im1, (x0, y0), (x1, y1), color, 4)

    plt.subplot(121), plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)), plt.title('Input image')
    plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)), plt.title('Rectified image')
    plt.axis('off')
    plt.show()


def compute_fundamintal_matrix(corr1, corr2, is_norm=True):

    # --- YOUR CODE HERE ---#
    # TODO compute the fundamental matrix F using the eight-points algorithm

    mean1 = np.mean(corr1, 0, dtype=float)
    mean2 = np.mean(corr2, 0, dtype=float)

    rmsd1 = np.mean(np.sqrt(np.power(corr1[:, 0] - mean1[0], 2) + np.power(corr1[:, 1] - mean1[1], 2)))
    rmsd2 = np.mean(np.sqrt(np.power(corr2[:, 0] - mean2[0], 2) + np.power(corr2[:, 1] - mean2[1], 2)))

    if is_norm:
        ncorr1 = (corr1 - mean1) * np.sqrt(2) / rmsd1
        ncorr2 = (corr2 - mean2) * np.sqrt(2) / rmsd2
    else:
        ncorr1 = corr1
        ncorr2 = corr2

    A = np.zeros((ncorr1.shape[0], 9), dtype=float)
    A[:, 0] = ncorr1[:, 0] * ncorr2[:, 0]
    A[:, 1] = ncorr1[:, 1] * ncorr2[:, 0]
    A[:, 2] = ncorr2[:, 0]
    A[:, 3] = ncorr1[:, 0] * ncorr2[:, 1]
    A[:, 4] = ncorr1[:, 1] * ncorr2[:, 1]
    A[:, 5] = ncorr2[:, 1]
    A[:, 6] = ncorr1[:, 0]
    A[:, 7] = ncorr1[:, 1]
    A[:, 8] = 1

    U, s, V = np.linalg.svd(A)
    f = V[8, :].reshape(3, 3)
    U, s, V = np.linalg.svd(f)

    ## enforce rank 2
    s_new = np.zeros((3, 3), dtype=float)
    s_new[0, 0] = s[0]
    s_new[1, 1] = s[1]

    f_rank2 = np.dot(U, np.dot(s_new, V))

    if is_norm:
        Tl = np.array([np.sqrt(2) / rmsd1, 0, -1 * mean1[0] * np.sqrt(2) / rmsd1, 0, np.sqrt(2) / rmsd1,
                       -1 * mean1[1] * np.sqrt(2) / rmsd1, 0, 0, 1]).reshape((3,3))
        Tr = np.array([np.sqrt(2) / rmsd2, 0, -1 * mean2[0] * np.sqrt(2) / rmsd2, 0, np.sqrt(2) / rmsd2,
              -1 * mean2[1] * np.sqrt(2) / rmsd2, 0, 0, 1]).reshape((3,3))
        f_rank2 = np.dot(Tr.transpose(), np.dot(f_rank2, Tl))

    f_rank2 = f_rank2 / f_rank2[-1, -1]

    return f_rank2


def task_06():

    # --- YOUR CODE HERE ---#
    # TODO read the images and the coordinates

    im_1 = cv2.imread(r'./data/Uni_Bonn_01.JPG')
    im_2 = cv2.imread(r'./data/Uni_Bonn_02.JPG')

    corr_all = np.genfromtxt(r'./data/corr.txt', dtype=float)
    corr_1 = corr_all[:, :2]
    corr_2 = corr_all[:, 2:]

    # --- YOUR CODE HERE ---#
    # TODO compute the fundamental matrix F using the eight-points algorithm

    F = compute_fundamintal_matrix(corr_1, corr_2)

    # --- YOUR CODE HERE ---#
    # TODO check the result of computing F
    # F_cv, _ = cv2.findFundamentalMat(corr_1, corr_2, cv2.FM_8POINT)
    # print('Check: ', F - F_cv)
    corr_1_w = np.concatenate((corr_1, np.ones(20)[:, None]), axis=1)
    corr_2_w = np.concatenate((corr_2, np.ones(20)[:, None]), axis=1)
    for ii in range(20):
        error = np.matmul(np.matmul(corr_2_w[ii, :], F), corr_1_w[ii, :].T)
        print('Epipolar error for point %s: %.2f pix' % (ii, error))

    # --- YOUR CODE HERE ---#
    # TODO draw and visualize the corresponding points and the draw the Epipolar lines

    drawEpipolar(im_1, im_2, corr_1, corr_2, F)

    # --- YOUR CODE HERE ---#
    # TODO rectify the images and visualize them with the corresponding points and Epipolar lines

    [_, H1, H2] = cv2.stereoRectifyUncalibrated(corr_1, corr_2, F, im_1.shape[:2])

    im1_dst = cv2.warpPerspective(im_1, H1, (im_1.shape[1]+300, im_1.shape[0]+300))  # wraped image
    im1_dst[:, :, [0, 1, 2]] = im1_dst[:, :, [2, 1, 0]]

    im2_dst = cv2.warpPerspective(im_2, H2, (im_2.shape[1]+300, im_2.shape[0]+300))  # wraped image
    im2_dst[:, :, [0, 1, 2]] = im2_dst[:, :, [2, 1, 0]]

    plt.subplot(121), plt.imshow(im2_dst), plt.title('Uni_Bonn_02')
    plt.axis('off')
    plt.subplot(122), plt.imshow(im1_dst), plt.title('Uni_Bonn_01')
    plt.axis('off')
    plt.show()



if __name__ == "__main__":

   task_06()