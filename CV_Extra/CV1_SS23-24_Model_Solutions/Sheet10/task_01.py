import numpy as np
import cv2

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def click_event(event, x, y, flags, param):

    # --- YOUR CODE HERE ---#
    # TODO extract x, y coordinate from a mouse event (click)

    global x_click, y_click
    if event == cv2.EVENT_LBUTTONDBLCLK:
        x_click, y_click = x, y
        print(x, y)


def extract_points(image):

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', click_event)

    pts_x, pts_y = [], []

    # --- YOUR CODE HERE ---#
    # TODO get the coordinates from a mouse event (click) for n_points and return the results in an array [n_points, 2]
    # TODO mark/plot the selected points with IDs on the image

    print('left click to pick a point and press (a) to save the point')

    p_nr = 1
    while True:
        cv2.imshow('image', image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # Esc key to stop
            break
        elif k == ord('a'):
            pts_x.append(x_click)
            pts_y.append(y_click)
            print("adding point: x={}, y={}".format(x_click, y_click))
            markerType = cv2.MARKER_CROSS
            markerSize = 40
            thickness = 4
            cv2.drawMarker(image, (x_click, y_click), (0, 255, 0), markerType, markerSize, thickness)
            cv2.putText(image, str(p_nr), (x_click+10, y_click), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
            p_nr += 1
            #cv2.circle(image, (x_click, y_click), 4, (0, 255, 0), 2)
    # close the window
    cv2.destroyAllWindows()

    return np.array([pts_x, pts_y]).T


def compute_Homography(pts_src, pts_target):

    # --- YOUR CODE HERE ---#
    # TODO compute Homography from a set of corresponding points

    Npts = len(pts_src)
    print("Number of measured points: ", Npts)
    # Generate homogeneous coordinates
    x_src = np.append(pts_src, np.ones([Npts, 1]), 1)
    x_target = np.append(pts_target, np.ones([Npts, 1]), 1)

    # Calculate the perspective transformation H

    A = np.zeros((2 * Npts, 9))

    for n in range(Npts):
        X_T = x_src[n, :].T
        x = x_target[n, 0]
        y = x_target[n, 1]
        w = x_target[n, 2]

        A[2 * n + 1, 3:6] = -w * X_T
        A[2 * n + 1, 6:] = y * X_T

        A[2 * n, :3] = w * X_T
        A[2 * n, 6:] = -x * X_T

    u, s, v = np.linalg.svd(A)

    # Solution to H is the last column of V, or last row of V transpose

    H = v[8].reshape((3, 3))
    H = H / H[-1, -1]

    return H


def check_Homography(pts_src, pts_target, H):

    # --- YOUR CODE HERE ---#
    # TODO check the Homography transformation and print the differences

    Npts = len(pts_src)
    x_src = np.append(pts_src, np.ones([Npts, 1]), 1)
    x_target = np.append(pts_target, np.ones([Npts, 1]), 1)

    x_cal = np.matmul(H, x_src.T).T

    x_cal[:, 0] = x_cal[:, 0] / x_cal[:, 2]
    x_cal[:, 1] = x_cal[:, 1] / x_cal[:, 2]
    x_cal[:, 2] = x_cal[:, 2] / x_cal[:, 2]

    # Compute the difference between transformed and original points

    dif = x_cal - x_target

    for p in range(Npts):
        pts_nr = str(p + 1) if p > 8 else '0' + str(p + 1)
        dif_x, dif_y = dif[p, :-1]

        print('transformation error for point %s: dif_x %.2f - dif_y %.2f pix' % (pts_nr, dif_x, dif_y))

def task_01():

    # --- YOUR CODE HERE ---#
    # TODO read the images and extract coordinates of corresponding points
    image_src = cv2.imread(r'.\data\Building_facade_02.JPG')

    pts_src = extract_points(image_src.copy())
    #pts_src = np.array([[818, 848], [768, 1747], [1147, 861], [1132, 1750], [1309, 1003],
    #                   [1309, 1283], [1477, 1006], [1484, 1287], [1629, 874], [1664, 1753]])

    image_target = cv2.imread(r'.\data\Building_facade_01.JPG')
    pts_target = extract_points(image_target.copy())

    #pts_target = np.array([[1232, 552], [1215, 1538], [1561, 643], [1574, 1592], [1722, 832], [1731, 1128], [1879, 870],
    #                       [1892, 1163], [2015, 766], [2065, 1665]])

    # --- YOUR CODE HERE ---#
    # TODO cmpute homography and check it
    H = compute_Homography(pts_src, pts_target)

    check_Homography(pts_src, pts_target, H)

    # --- YOUR CODE HERE ---#
    # TODO stitch the images and visualize the result

    dst = cv2.warpPerspective(image_src, H, ((image_src.shape[1] + image_target.shape[1]), image_target.shape[0]))
    dst[0:image_target.shape[0], 0:image_target.shape[1]] = image_target

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', dst[:, :, :])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

   task_01()

