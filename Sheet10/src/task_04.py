import numpy as np

np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format})

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def extract_from_P(P):

    # --- YOUR CODE HERE ---#
    # TODO decompose P into the camera extrinsic and intrinsic parameters

    return


def construct_P(f, mx, my, px, py, R, X0):

    # --- YOUR CODE HERE ---#
    # TODO define a projection matrix P using interior and exterior orientation parameters

    return P


def task_04():

    P = np.array([[-14066.493447, -9945.538111, 4796.664032, 8463868675.444821],
                  [9900.140035, -14625.938658, 590.458246, -176902714.499046],
                  [0.113060, -0.051300, 0.992263, -33765.746644]])

    # --- YOUR CODE HERE ---#
    # TODO decompose P and print parameters

   # = extract_from_P(P)

    # --- YOUR CODE HERE ---#
    # TODO reconstruct P and print the result and differences
    #P_rec = construct_P(f, mx, my, px, py, R, X0)

    # --- YOUR CODE HERE ---#
    # TODO compute ground sampling distance and camera footprint


if __name__ == "__main__":

    task_04()

