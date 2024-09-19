import numpy as np
import scipy
from pr3_utils import *

def projDeriv(q):
	# potential divide by zero
	q1, q2, q3, q4 = q[0], q[1], q[2], q[3]

	prod = (1.0)/q3
	arr = np.array([[1, 0, (-1*q1)/q3, 0], [0, 1, (-1*q2)/q3, 0], [0, 0, 0, 0], [0, 0, (-1*q4)/q3, 1]])
	return prod*arr

def homo(m):
	return np.hstack((m, np.array([1])))

def hatMap(x):
    return np.array([[0, -1*x[2], x[1]], [x[2], 0, -1*x[0]], [-1*x[1], x[0], 0]])

def uHat(u):
	w = u[3:]
	v = u[:3]

	w_hat = hatMap(w)

	return np.array([[w_hat[0][0], w_hat[0][1], w_hat[0][2], v[0]], [w_hat[1][0], w_hat[1][1], w_hat[1][2], v[1]], [w_hat[2][0], w_hat[2][1], w_hat[2][2], v[2]], [0, 0, 0, 0]])

def uCurly(u):
	w = u[3:]
	v = u[:3]
	
	w_hat = hatMap(w)
	v_hat = hatMap(v)
	zeros = np.array([0, 0, 0])

	line1 = np.hstack((w_hat[0], v_hat[0]))
	line2 = np.hstack((w_hat[1], v_hat[1]))
	line3 = np.hstack((w_hat[2], v_hat[2]))
	line4 = np.hstack((zeros, w_hat[0]))
	line5 = np.hstack((zeros, w_hat[1]))
	line6 = np.hstack((zeros, w_hat[2]))

	return np.array([line1, line2, line3, line4, line5, line6])

def proj(q):
	q3 = q[2]
	return ((1.0)/q3) * q

def stereoModel(zt, fsu, fsv, cu, cv, b):
	uL = zt[0]
	vL = zt[1]
	uR = zt[2]

	d = uL - uR
	z = (fsu*b)/d
	x = (z*(uL - cu))/fsu
	y = (z*(vL - cv))/fsu
	
	return np.array([x, y, z, 1])


if __name__ == '__main__':

	# Load the measurements
	filename = "../data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

	# (a) IMU Localization via EKF Prediction
	# obtain initial time stamp
	prev_t = t[0][0]

	# set initial covariance and perturbations
	W = 0.01*np.identity(6)
	prev_cov = 0.01*np.identity(6)
	mean = np.array([0, 0, 0, 0, 0, 0])
	prev_delta = np.random.multivariate_normal(mean, prev_cov, 1)[0]

	# set initial mean
	prev_mew = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

	# set initial pose
	deltaMap = uHat(prev_delta)
	T_prev = np.matmul(prev_mew, deltaMap)

	poses = []
	poses.append(prev_mew)
	
	for i in range(1,len(t[0])):
		# append the mean for visualization
		

		tau = t[0][i] - prev_t
		v = linear_velocity[:, i]
		w = angular_velocity[:, i]
		u = np.hstack((v, w))
		u_hat = uHat(u)

		next_mew = np.matmul(prev_mew, scipy.linalg.expm(tau*u_hat))

		# do stuff with mew here:
		cov = np.matmul(np.matmul(scipy.linalg.expm(-1*tau*uCurly(u)), prev_cov), scipy.linalg.expm(-1*tau*uCurly(u).T)) + W
		w_noise = np.random.multivariate_normal(mean, W, 1)[0]
		delta = np.matmul(scipy.linalg.expm(-1*tau*uCurly(u)), prev_delta) + w_noise
		
		deltaMap = uHat(delta)
		# T = np.matmul(next_mew, deltaMap)

		# T_prev = T
		prev_delta = delta
		prev_cov = cov
		prev_mew = next_mew
		poses.append(prev_mew)
		prev_t = t[0][i]
	
	np_means = np.array(poses)
	reshaped_means = np.transpose(np_means, [1,2,0])

	# TODO: uncomment this
	# visualize_trajectory_2d(reshaped_means, show_ori = True)
	
	# (b) Landmark Mapping via EKF Update
	# len(t) = 3026 elements
	# Nt = 13289

	# constants
	P = np.hstack((np.identity(3), np.array([[0], [0], [0]])))
	fsu = K[0][0]
	cu = K[0][2]
	fsv = K[1][1]
	cv = K[1][2]

	# calibration matrix
	rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	Ks = np.array([[fsu, 0, cu, 0], [0, fsv, cv, 0], [fsu, 0, cu, -fsu*b], [0, fsv, cv, 0]])
	imu_T_Cam = rot @ imu_T_cam
	O_T_I = np.linalg.inv(imu_T_cam) # we want IMU to camera 

	# for every time step, loop through every observation and check for invalid observations
	stride = 30

	strided_features = np.empty((4, len(features[:, :, 0][0][::stride]), len(t[0])), dtype=float)
	for i in range(0, len(t[0])):
		for j in range(4):
			strided_features[:, :, i][j] = features[:, :, i][j][::stride]

	M = int(len(strided_features[:, :, 1][0])) # number of landmarks

	# init the V and I
	V = np.identity(4)

	# init state and covs
	prev_mew = np.zeros(3*M)
	prev_cov = np.identity(3*M)

	# visited map
	observations = strided_features[:, :, 0]
	W_T_I = poses[0] # IMU to world
	observedIndices = np.where(np.sum(observations, axis=0) != -4)[0]
	isVisited = [] # {new_item: False for new_item in observedIndices}

	# initialize timestep 0
	N = len(observedIndices) 
	for j in range(0, N): # downsample the number of observations
		isVisited.append(observedIndices[j])
		zt = observations[:, observedIndices[j]]
		coord = stereoModel(zt, fsu, fsv, cu, cv, b) # optical coordinates
		
		robo_coord = imu_T_cam @ coord # rot @ coord
		prev_mew[3*observedIndices[j]:3*observedIndices[j]+3] = robo_coord[:3]
	
	print(prev_mew)
	print("init is done")

	# stride the time for faster compute
	for i in range(1, len(t[0])):
		observations = strided_features[:, :, i] # valid observations
		W_T_I = poses[i] # IMU to world

		observedIndices = np.where(np.sum(observations, axis=0) != -4)[0]
		# remove observations observed for the first time and initialize them
		toDelete = []
		for j in range(len(observedIndices)):
			if observedIndices[j] not in isVisited:
				zt = observations[:, observedIndices[j]]
				coord = stereoModel(zt, fsu, fsv, cu, cv, b)
				robo_coord = W_T_I @ imu_T_cam @ coord
				prev_mew[3*observedIndices[j]:3*observedIndices[j]+3] = robo_coord[:3]

	visualize_trajectory_2d2(reshaped_means, prev_mew)
	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)