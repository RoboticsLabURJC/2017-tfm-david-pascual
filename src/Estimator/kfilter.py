import numpy as np
import pykalman as pk


class KFilter3D:
    def __init__(self):
        self.measurements = None
        self.kf = None

    def init_filter(self, init_pos, fps=30.):
        x, y, z = init_pos
        initial_state_mean = [x, 0, y, 0, z, 0]

        transition_matrix = [[1, 1 / fps, 0,       0, 0,       0],
                             [0,       1, 0,       0, 0,       0],
                             [0,       0, 1, 1 / fps, 0,       0],
                             [0,       0, 0,       1, 0,       0],
                             [0,       0, 0,       0, 1, 1 / fps],
                             [0,       0, 0,       0, 0,       1]]

        observation_matrix = [[1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0]]

        self.kf = pk.KalmanFilter(transition_matrices=transition_matrix,
                                  observation_matrices=observation_matrix, )
        self.x_now = initial_state_mean
        self.P_now = np.eye(6, 6)
        self.measurements = np.array([[x, y, z]])
        self.prev_measurement = np.ones_like(self.measurements[-1]) * np.nan

        self.lost=False

    # def update_filter(self, measurement, mean_pos, th_dist_per_frame=500, th_dist_to_center=1500):
    #     d_mean = np.sqrt(np.sum(np.square(measurement - mean_pos)))
    #
    #     if self.measurements is None and d_mean < th_dist_to_center:
    #         self.init_filter(measurement)
    #     if self.lost and not np.isnan(measurement).all() and d_mean < th_dist_to_center:
    #         self.init_filter(measurement)
    #
    #     prev_measurement = self.measurements[-1]
    #     measurement = np.array(measurement).reshape((1, 3))
    #     d_pos = np.sqrt(np.sum(np.square(measurement - prev_measurement)))
    #     if d_pos > th_dist_per_frame or d_mean > th_dist_to_center or any(np.isnan(measurement.squeeze())):
    #         measurement = np.array([[np.nan, np.nan, np.nan]])
    #
    #     self.measurements = np.append(self.measurements, measurement, axis=0)
    #
    #     nan_in_last_frames = np.isnan(self.measurements[-3:])
    #     if np.count_nonzero(nan_in_last_frames) >= np.count_nonzero(~nan_in_last_frames):
    #         self.lost = True
    #         return self.x_now[0], self.x_now[2], self.x_now[4]
    #
    #     if any(np.isnan(measurement.squeeze())):
    #         measurement = np.ma.masked
    #     else:
    #         self.measurements_em = np.append(self.measurements_em, measurement, axis=0)
    #         measurement = list(measurement.squeeze())
    #
    #     (self.x_now, self.P_now) = self.kf.filter_update(filtered_state_mean=self.x_now,
    #                                                      filtered_state_covariance=self.P_now,
    #                                                      observation=measurement)
    #
    #     return self.x_now[0], self.x_now[2], self.x_now[4]


    def update_filter(self, measurement, mean_pos, th_dist_per_frame=500, th_dist_to_center=1500):
        d_mean = np.sqrt(np.sum(np.square(measurement - mean_pos)))

        if self.measurements is None or self.lost:
            if not np.isnan(measurement).all() and d_mean < th_dist_to_center:
                self.init_filter(measurement)

            if self.measurements is not None:
                return self.x_now[0], self.x_now[2], self.x_now[4]
            else:
                return measurement

        else:
            self.prev_measurement = np.array([self.x_now[0], self.x_now[2], self.x_now[4]])

            d_pos = np.sqrt(np.sum(np.square(measurement - self.prev_measurement)))

            if d_pos > th_dist_per_frame or d_mean > th_dist_to_center or any(np.squeeze(np.isnan(measurement))):
                measurement = np.array([[np.nan, np.nan, np.nan]])
            else:
                measurement = np.array(measurement).reshape((1, 3))

            self.measurements = np.append(self.measurements, measurement, axis=0)

            nan_in_last_frames = np.isnan(self.measurements[-3:])
            if np.count_nonzero(nan_in_last_frames) >= np.count_nonzero(~nan_in_last_frames):
                self.lost = True
                return self.x_now[0], self.x_now[2], self.x_now[4]

            measurement = measurement.squeeze()
            if any(np.isnan(measurement)):
                measurement = np.ma.masked

            (self.x_now, self.P_now) = self.kf.filter_update(filtered_state_mean=self.x_now,
                                                             filtered_state_covariance=self.P_now,
                                                             observation=measurement)

            return self.x_now[0], self.x_now[2], self.x_now[4]
