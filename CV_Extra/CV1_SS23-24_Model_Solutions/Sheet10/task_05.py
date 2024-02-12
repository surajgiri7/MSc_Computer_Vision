import numpy as np
import cv2
import glob

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def task_05():

    images = glob.glob(r'./camera_calibration/*.JPG')

    # --- YOUR CODE HERE ---#
    # TODO calibrate the camera with checkerboard patterns

    objp = np.zeros((10*12, 3), np.float32)
    objp[:, :2] = np.mgrid[0:12, 0:10].T.reshape(-1, 2)

    # scale the object
    objp = objp * 12

    objpoints = []
    imgpoints = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #print(fname)
        ret, corners = cv2.findChessboardCorners(gray, (12, 10), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
            imgpoints.append(corners2)

         #   img = cv2.drawChessboardCorners(img, (12, 10), corners2, ret)
         #   cv2.namedWindow('img', cv2.WINDOW_NORMAL)
         #   cv2.imshow('img', img)
         #   cv2.waitKey(0)
         #   cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # --- YOUR CODE HERE ---#
    # TODO print the results of the calibration in a readable way

    print('Camera matrix: ', mtx)
    print('Distortion coefficients: k1 %.4f k2 %.4f p1 %.4f p2 %.4f k3 %.4f'
          % (dist[0, 0], dist[0, 1], dist[0, 2], dist[0, 3], dist[0, 4]))

    focal_length = mtx[1, 1] * (1.4 * 10**-3)
    print('focal length', focal_length)

    # --- YOUR CODE HERE ---#
    # TODO compute the re-projection error

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        #error = np.sqrt(np.sum(np.multiply(diff, diff)) / len(imgpoints2))
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print('total error: %.4f pix' % (mean_error/len(objpoints)))


if __name__ == "__main__":

    task_05()

