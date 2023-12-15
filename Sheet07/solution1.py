import numpy as np
import matplotlib.pylab as plt

observations = np.load('data/observations.npy')


def get_observation(t):
    return observations[t]

def get_data_at_iteration(n, datalist):
    x_1 = [x[0] for x in datalist[:n+1]]
    y_1 = [x[1] for x in datalist[:n+1]]

    return x_1, y_1 


class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.psi = psi
        self.sigma_p = sigma_p
        self.phi = phi
        self.sigma_m = sigma_m
        self.state = None
        self.convariance = None
        self.tau = tau

    def init(self, init_state):
        self.state = init_state
        self.covariance = np.identity(init_state.shape[0]) * 0.01
        pass

    def track(self, xt):
        # State prediction
        state_pred = np.dot(self.psi, self.state)
        # Covariance prediction
        covariance_pred = self.sigma_p + np.dot(np.dot(self.psi, self.covariance), np.transpose(self.psi))
        # Kalman Gain
        kalman_gain1 = np.dot(covariance_pred, np.transpose(self.phi))
        kalman_gain2 = np.dot(np.dot(self.phi, covariance_pred), np.transpose(self.phi))
        kalman_gain3 = np.linalg.inv(self.sigma_m + kalman_gain2)
        kalman_gain = np.dot(kalman_gain1, kalman_gain3)
        # State update
        state_update = state_pred + np.dot(kalman_gain, (xt - np.dot(self.phi, state_pred)))
        # Covariance update
        covariance_update = np.dot((np.identity(kalman_gain.shape[0]) - np.dot(kalman_gain, self.phi)), covariance_pred)
        self.state = state_update
        self.covariance = covariance_update
        pass

    def get_current_location(self):
        return self.phi @ self.state

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

    init_state = np.array([0, 1, 0, 0])

    psi, sigma_p, phi, sigma_m = get_world_model()


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    tracker = KalmanFilter(psi, sigma_p, phi, sigma_m, 1)
    tracker.init(init_state)

    track = perform_tracking(tracker)

    num_steps = len(observations) 
    for n in range(num_steps):
        ax.clear()
        o_x, o_y = get_data_at_iteration(n, observations) 
        # TODO: 
        t_x, t_y = get_data_at_iteration(n, track) 
        # t_x, t_y = 0, 0
         
        ax.plot(o_x, o_y, 'g', label='observations')
        ax.plot(t_x, t_y, 'y', label='Kalman')
        ax.legend()
        plt.pause(0.01)
    # plt.pause(3)
    plt.show()

if __name__ == "__main__":
    main()
