from pykalman import KalmanFilter
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import time

measurements = np.array([[399,293, 234],[403,299, 245],[409,308, 256],[416,315, 245],[418,318, 245],[420,323, 234],[429,326, 236],[423,330, 240],[429,334, 240],[431,337, 245],[433,342, 250],[434,352, 251],[434,349, 241],[433,350, 235],[431,350, 231],[430,349, 226],[428,347, 223],[427,345, 218],[425,341, 220],[429,338, 225],[431,328, 228],[410,313, 260],[406,306, 232],[402,299, 237],[397,291, 240],[391,294, 243],[376,270, 245],[372,272, 250],[351,248, 260],[336,244, 270],[327,236, 275],[307,220, 277]])

print(measurements.shape)
initial_state_mean = [measurements[0, 0],
                      0,
                      measurements[0, 1],
                      0,
                      measurements[0, 2],
                      0]

transition_matrix = [[1, 1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 1]]

observation_matrix = [[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0]]

kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)

measurements = np.ma.array(measurements)
for t in range(measurements.shape[0]):
    if t % 4 != 0:
        measurements[t] = np.ma.masked

kf1 = kf1.em(measurements, n_iter=5)

time_before = time.time()

kf3 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean,
                  observation_covariance = np.ones((3, 3)) * 7)

n_real_time = 10
# kf3 = kf3.em(measurements[:-n_real_time, :], n_iter=5)
# (filtered_state_means, filtered_state_covariances) = kf3.filter(measurements[:-n_real_time,:])

print("Time to build and train kf3: %s seconds" % (time.time() - time_before))

x_now = initial_state_mean
P_now =  np.eye(6, 6)
x_new = np.zeros((n_real_time, 6))
i = 0

for measurement in measurements[-n_real_time:, :]:
    time_before = time.time()
    (x_now, P_now) = kf3.filter_update(filtered_state_mean = x_now,
                                       filtered_state_covariance = P_now,
                                       observation = measurement)
    print("Time to update kf3: %s seconds" % (time.time() - time_before))
    x_new[i, :] = x_now
    i = i + 1

plt.figure(3)
times = range(measurements.shape[0])
old_times = range(measurements.shape[0] - n_real_time)
new_times = range(measurements.shape[0]-n_real_time, measurements.shape[0])
# plt.plot(times, measurements[:, 0], 'bo',
#          times, measurements[:, 1], 'ro',
#          times, measurements[:, 2], 'go',
#          old_times, filtered_state_means[:, 0], 'b--',
#          old_times, filtered_state_means[:, 2], 'r--',
#          old_times, filtered_state_means[:, 4], 'g--',
plt.plot(new_times, x_new[:, 0], 'b-',
         new_times, x_new[:, 2], 'r-',
         new_times, x_new[:, 4], 'g-')

plt.show()