import numpy as np

np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format})

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def extract_P(P):
    # Extract X0, R, K from a projection matrix P e.g.obtained by DLT

    # --- YOUR CODE HERE ---#
    # TODO decompose P into the camera extrinsic and intrinsic parameters

    # Input
    # p: projection matrix
    # Output
    # X0: translation vector - coordinates of principal point in object frame [m]
    # R: rotation matrix world to image [Radian]
    # k: Camera matrix [pixel]

    # Algorithm
    # Decompose P matrix
    U, D, V = np.linalg.svd(P)
    # translation vector last column of V
    X0 = V[-1]
    X0 = X0[:-1] / X0[-1]

    # left 3x3 submatrix of p
    M = P[:3, :3]

    # QR factorization returns an upper triangular matrix R and a unitary matrix Q, where M = Q * R
    Q, R = np.linalg.qr(np.linalg.inv(M))

    # camera matrix
    K = np.linalg.inv(R)
    K = K / K[-1, -1]

    # rotation matrix
    R = Q.T

    return X0, R, K


def construct_P(f, mx, my, px, py, R, X0):

    # --- YOUR CODE HERE ---#
    # TODO define a projection matrix P using interior and exterior orientation parameters

    # Input
    # f: focal length [mm]
    # mx: pixel size in horizontal direction [mm]
    # my: pixel size in vertical direction [mm]
    # px: x coordinate of principal point in image frame [pixel]
    # py: y coordinate of principal point in image frame [pixel]
    # R: rotation matrix world to image [Radian]
    # X0: translation vector - coordinates of principal point in object frame [m]

    # Output P: Projection Matrix enable a transformation between object and pixel coordinates

    # --------------------------------------------------------------

    # Algorithm

    K = np.array([[f[0] * mx, 0, px],
                  [0, f[1] * my, py],
                  [0, 0, 1]])
    Rt = np.hstack((R, np.matmul(-R, X0)[:, None]))
    P = np.matmul(K, Rt)

    return P


def task_04():

    P = np.array([[-14066.493447, -9945.538111, 4796.664032, 8463868675.444821],
                  [9900.140035, -14625.938658, 590.458246, -176902714.499046],
                  [0.113060, -0.051300, 0.992263, -33765.746644]])

    inv_mx, inv_my = 0.002, 0.002

    # --- YOUR CODE HERE ---#
    # TODO decompose P

    X0, R, K = extract_P(P)

    f_x = K[0, 0] * inv_mx
    f_y = K[1, 1] * inv_my

    print('Translation vector: ', X0)
    print('Rotation Matrix: ', R)
    print('Camera Matrix: ', K)
    print('Focal length x:', f_x)
    print('Focal length y:', f_y)

    # --- YOUR CODE HERE ---#
    # TODO construct P
    P_rec = construct_P((f_x, f_y), 1 / inv_mx, 1 / inv_my, K[0, -1], K[1, -1], R, X0)
    print()
    print('diff: ', P - P_rec)
    print()

    # --- YOUR CODE HERE ---#
    # TODO compute ground sampling distance and camera footprint

    image_width = 7360
    image_height = 4912

    H = 100

    GSD_x = (inv_mx * H * 100) / f_x
    GSD_y = (inv_my * H * 100) / f_y

    print('GSD_x', GSD_x)
    print('GSD_y', GSD_y)

    Dw = (GSD_x*image_width) / 100
    Dh = (GSD_y*image_height) / 100

    print('\nfootprint_w', Dw)
    print('footprint_h', Dh)


if __name__ == "__main__":
    task_04()

