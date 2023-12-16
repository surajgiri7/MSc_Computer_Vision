import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):

        self.size = psi.shape[0] * (tau + 1)
        self.psi = np.zeros((self.size,self.size))
        ident = np.identity(self.size - psi.shape[0])
        self.psi[0:4,0:4] = psi
        self.psi[4:,0:-4] = ident
        self.psi[3,3] *= tau
        self.psi[2,2] *= tau


        self.sigma_p = np.zeros((self.size, self.size))
        self.sigma_p[0:4,0:4] = sigma_p

        self.sigma_m = sigma_m

        self.phi = np.zeros((2,self.size))
        self.phi[0:2,0:4] = phi


        self.state = None
        self.convariance = None
        self.tau = tau
        #self.sigma_t = sigma_p

    def init(self, init_state):
        self.state = np.zeros(self.size)
        self.state[0:4] = init_state
        self.covariance = np.identity(self.size)
        pass

    def track(self, xt):
        #State prediction
        mue_prediction = np.dot(self.psi, self.state)
        print("mue_prediction", mue_prediction)
        #Covariance prediction
        sigma_p_prediction = self.sigma_p + np.dot(np.dot(self.psi, self.covariance), np.transpose(self.psi))
        print("sigma_p_prediction", sigma_p_prediction)
        #Kalman Gain
        kalman_gain1 = np.dot(sigma_p_prediction, np.transpose(self.phi))
        kalman_gain2 = np.dot(np.dot(self.phi, sigma_p_prediction), np.transpose(self.phi))
        kalman_gain3 = np.linalg.inv(self.sigma_m + kalman_gain2)
        kalman_gain = np.dot(kalman_gain1, kalman_gain3)
        print("kalman_gain", kalman_gain)
        #State update
        inner = xt - np.dot(self.phi, mue_prediction)
        mu_t = mue_prediction + np.dot(kalman_gain, inner)
        print("mu_t", mu_t)
        #Covariance update
        sigma_t = np.dot(np.identity(self.size) - np.dot(kalman_gain, self.phi), sigma_p_prediction)
        print('sigma_t', sigma_t)

        self.covariance = sigma_t
        self.state = mu_t
        pass

    def get_current_location(self):
        return self.state[-4:-2]
        pass

def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    return track

def main():
    init_state = np.array([0, 1, 0, 0])

    psi = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    phi = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
    sm = 0.05
    sigma_m = np.array([[sm, 0],
                        [0, sm]])


    tracker = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=0)


if __name__ == "__main__":
    main()