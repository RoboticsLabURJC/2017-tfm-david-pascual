import numpy as np
import pykalman as pk


class KFilter3D:
    def __init__(self):
        self.measurements = None


    def init_filter(self, init_pos):
        x, y, z = init_pos
        initial_state_mean = [x, 0, y, 0, z, 0]

        transition_matrix = [[1, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1],
                             [0, 0, 0, 0, 0, 1]]

        observation_matrix = [[1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0]]

        self.kf = pk.KalmanFilter(transition_matrices=transition_matrix,
                                  observation_matrices=observation_matrix,
                                  initial_state_mean=initial_state_mean,
                                  observation_covariance=np.ones((3, 3)) * 1)
        self.x_now = initial_state_mean
        self.P_now = np.eye(6, 6)
        self.measurements = np.array([[x, y, z]])

        self.lost = False

    def update_filter(self, measurement):
        if self.measurements is None:
            self.init_filter(measurement)
        if self.lost and not np.isnan(measurement).all():
            self.init_filter(measurement)

        measurement = np.array(measurement).reshape((1, 3))
        # diff = np.sqrt(np.sum(np.square(measurement - self.measurements[-1])))
        # print("diff", diff)
        # if diff > 100:
        #     measurement = np.array([np.nan, np.nan, np.nan]).reshape((1, 3))

        self.measurements = np.append(self.measurements, measurement, axis=0)

        if np.isnan(self.measurements[-3:]).all():
            self.lost = True
            return np.nan, np.nan, np.nan

        if any(np.isnan(measurement.squeeze())):
            measurement = np.ma.masked
        else:
            measurement = list(measurement.squeeze())
        print("kInput", measurement)
        (self.x_now, self.P_now) = self.kf.filter_update(filtered_state_mean=self.x_now,
                                                         filtered_state_covariance=self.P_now,
                                                         observation=measurement)

        return self.x_now[0], self.x_now[2], self.x_now[4]
