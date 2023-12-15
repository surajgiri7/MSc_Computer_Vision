import numpy as np
import matplotlib.pylab as plt

observations = np.load('./data/observations.npy')


def get_observation(t):
    return observations[t]

def get_data_at_iteration(n, datalist):
    x_1 = [x[0] for x in datalist[:n+1]]
    y_1 = [x[1] for x in datalist[:n+1]]

    return x_1, y_1 

class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.tau = tau
        self.size = psi.shape[0] * (tau + 1)

        self.psi = self._initialize_psi(psi)
        self.sigma_p = self._initialize_sigma_p(sigma_p)
        self.phi = self._initialize_phi(phi)
        self.sigma_m = sigma_m

        self.state = None
        self.convariance = None

    def _initialize_psi(self, psi):
        psi_matrix = np.zeros((self.size, self.size))
        identity_mat = np.identity(self.size - psi.shape[0])

        psi_matrix[0:4, 0:4] = psi
        psi_matrix[4:, 0:-4] = identity_mat
        psi_matrix[3, 3] *= self.tau
        psi_matrix[2, 2] *= self.tau

        return psi_matrix

    def _initialize_sigma_p(self, sigma_p):
        sigma_p_matrix = np.zeros((self.size, self.size))
        sigma_p_matrix[0:4, 0:4] = sigma_p

        return sigma_p_matrix

    def _initialize_phi(self, phi):
        phi_matrix = np.zeros((2, self.size))
        phi_matrix[0:2, 0:4] = phi

        return phi_matrix

    def init(self, init_state):
        self.state = np.zeros(self.size)
        self.state[0:4] = init_state
        self.covariance = np.identity(self.size)

    def track(self, xt):
        # State prediction
        state_pred = np.dot(self.psi, self.state)
        # Covariance prediction
        sigma_p_prediction = self.sigma_p + np.dot(np.dot(self.psi, self.covariance), np.transpose(self.psi))
        # Kalman Gain
        kalman_gain = np.dot(np.dot(sigma_p_prediction, np.transpose(self.phi)), np.linalg.inv(self.sigma_m + np.dot(np.dot(self.phi, sigma_p_prediction), np.transpose(self.phi))))
        # Covariance update
        sigma_t = np.dot(np.identity(self.size) - np.dot(kalman_gain, self.phi), sigma_p_prediction)

        # State update
        self.covariance = sigma_t
        inner = xt - np.dot(self.phi, state_pred)
        self.state = state_pred + np.dot(kalman_gain, inner)

    def get_current_location(self):
        return self.state[-4:-2]


def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())
    return track

def get_world_model():
    psi = np.array([[1, 0, 1.5, 0],
                    [0, 1, 0, 0.5],
                    [0, 0, -1, 0],
                    [0, 0, 0, -2]])
    sp = 0.001
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp, 0],
                        [0, 0, 0, sp]])

    phi = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
    sm = 0.05
    sigma_m = np.array([[sm, 0],
                        [0, sm]])

    return psi, sigma_p, phi, sigma_m 


def main():
    psi, sigma_p, phi, sigma_m = get_world_model()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    fixed_lag_smoother = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=5)
    fixed_lag_smoother.init(np.array([0, 1, 0, 0]))

    track_smoothed = perform_tracking(fixed_lag_smoother)

    num_steps = len(observations)
    for n in range(num_steps):
        ax.clear()
        o_x, o_y = get_data_at_iteration(n, observations)
        s_x, s_y = get_data_at_iteration(n, track_smoothed)

        ax.plot(o_x, o_y, 'g', label='observations')
        ax.plot(s_x, s_y, 'b', label='Fixed Lag Smoother')
        ax.legend()
        plt.pause(0.01)
    plt.show()


if __name__ == "__main__":
    main()
