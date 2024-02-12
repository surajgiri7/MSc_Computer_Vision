import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    # TODO mark the selected points on the image

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
            cv2.putText(image, str(p_nr), (x_click + 10, y_click), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2,
                        cv2.LINE_AA)
            p_nr += 1
            # cv2.circle(image, (x_click, y_click), 4, (0, 255, 0), 2)
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


def task_02():
    # --- YOUR CODE HERE ---#
    # TODO read the image and extract the coordinates of corner points

    im_ori = cv2.imread(r'./data/Book.JPG')

    pts_src = extract_points(im_ori.copy())
    #pts_src = np.array([[1009, 2373], [2568, 1185], [198, 1575], [1193, 1009]])

    # --- YOUR CODE HERE ---#
    # TODO get pts_target

    im_height_new = round(19.7 * 30)  # 30 is a scale factor
    im_width_new = round(12.9 * 30)

    pts_target = np.float32([[0, 0], [0, im_width_new], [im_height_new, 0], [im_height_new, im_width_new]])

    # --- YOUR CODE HERE ---#
    # TODO cmpute homography
    H = compute_Homography(pts_src, pts_target)

    # --- YOUR CODE HERE ---#
    # TODO warp the image and visualize the result

    dst = cv2.warpPerspective(im_ori, H, (im_height_new, im_width_new))  # wrapped image

    im_ori[:, :, [0, 1, 2]] = im_ori[:, :, [2, 1, 0]]
    dst[:, :, [0, 1, 2]] = dst[:, :, [2, 1, 0]]

    dst = np.flip(np.swapaxes(dst, 0, 1), 0)

    plt.subplot(121), plt.imshow(im_ori), plt.title('Input image')
    plt.axis('off')
    plt.subplot(122), plt.imshow(dst), plt.title('Rectified image')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    task_02()

