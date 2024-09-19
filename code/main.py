import numpy as np
import scipy
from pr3_utils import *

# size n x n
def findTrace(mat, n):
 
    sum = 0
    for i in range(n):
        sum += mat[i][i]
    return sum

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

	return np.vstack((line1, line2, line3, line4, line5, line6))

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

def circleDot(hom_s):
	s_hat = hatMap(hom_s[:3])

	row1 = np.hstack((np.array([1, 1, 1]), -1*s_hat[0]))
	row2 = np.hstack((np.array([1, 1, 1]), -1*s_hat[1]))
	row3 = np.hstack((np.array([1, 1, 1]), -1*s_hat[2]))
	row = np.vstack((np.vstack((row1, row2)), row3))

	return np.vstack((row, np.array([0, 0, 0, 0, 0, 0])))

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

		# T_prev = T
		prev_delta = delta
		prev_cov = cov
		prev_mew = next_mew
		poses.append(prev_mew)
		prev_t = t[0][i]
	
	np_means = np.array(poses)
	reshaped_means = np.transpose(np_means, [1,2,0])

	visualize_trajectory_2d(reshaped_means, show_ori = True)
	
	# (b) Landmark Mapping via EKF Update
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
	stride = 20

	strided_features = np.empty((4, len(features[:, :, 0][0][::stride]), len(t[0])), dtype=float)
	for i in range(0, len(t[0])):
		for j in range(4):
			strided_features[:, :, i][j] = features[:, :, i][j][::stride]

	M = int(len(strided_features[:, :, 1][0])) # number of landmarks

	# init the V and I
	V = np.identity(4)
	
	# init state and covs
	prev_mew = np.zeros(3*M)
	prev_cov = 0.01 * np.identity(3*M)
	
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
				isVisited.append(observedIndices[j])
				toDelete.append(j)

		observedIndices = np.delete(observedIndices, toDelete)
				
		N = len(observedIndices)
		I = np.identity(N)

		# init the z and H stores relative to M
		ztildes = np.zeros(4*N)
		zs = np.zeros(4*N)
		Hs = np.zeros((4*N, 3*M))
		
		# predict step
		for j in range(N): # downsample the number of observations
			# compute ztilde
			arg = (O_T_I @ np.linalg.inv(W_T_I)) @ homo(prev_mew[3*observedIndices[j]:3*observedIndices[j]+3])
			ztilde = Ks @ proj(arg)
			ztildes[4*j:4*j+4] = ztilde
			zs[4*j:4*j+4] = observations[:, observedIndices[j]]

			# compute H
			arg = (O_T_I @ np.linalg.inv(W_T_I)) @ homo(prev_mew[3*observedIndices[j]:3*observedIndices[j]+3])
			H = (((Ks @ (projDeriv(arg))) @ O_T_I) @ np.linalg.inv(W_T_I)) @ P.T
			Hs[4*j:4*j+4,3*observedIndices[j]:3*observedIndices[j]+3] = H

		# update step 
		K_next = (prev_cov @ Hs.T) @ np.linalg.inv((Hs @ prev_cov) @ Hs.T + np.kron(I, V))
		mew_next = prev_mew + K_next @ (zs - ztildes)
		I_big = np.identity(len((K_next @ Hs)[0]))
		cov_next = (I_big - K_next @ Hs) @ prev_cov

		prev_mew = mew_next
		prev_cov = cov_next
	
	visualize_trajectory_2d2(reshaped_means, prev_mew)
	
	# (c) Visual-Inertial SLAM
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
	stride = 40

	strided_features = np.empty((4, len(features[:, :, 0][0][::stride]), len(t[0])), dtype=float)
	for i in range(0, len(t[0])):
		for j in range(4):
			strided_features[:, :, i][j] = features[:, :, i][j][::stride]

	M = int(len(strided_features[:, :, 1][0])) # number of landmarks
	# --------- init the covariance matrix, landmark mean, pose mean, and list of poses ---------
	prev_cov = np.identity(3*M + 6)
	prev_mew_robot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	prev_mew_landmarks = np.zeros(3*M)

	prev_cov[3*M+5, 3*M+5] = 0.001
	prev_cov[3*M+4, 3*M+4] = 0.001
	prev_cov[3*M+3, 3*M+3] = 0.001
	prev_cov[3*M+2, 3*M+2] = 0.01
	prev_cov[3*M+1, 3*M+1] = 0.01
	prev_cov[3*M, 3*M] = 0.01

	poses = []
	poses.append(prev_mew_robot)

	prev_t = t[0][0]

	# visited map
	observations = strided_features[:, :, 0]
	W_T_I = poses[0] # IMU to world
	observedIndices = np.where(np.sum(observations, axis=0) != -4)[0]
	isVisited = [] # {new_item: False for new_item in observedIndices}

	# initialize timestep 0 landmarks
	N = len(observedIndices) 
	I = np.identity(N)
	V = 5 * np.identity(4)
	W = 0.01*np.identity(6) # last 3 use a lower value
	W[3,3] = 0.0001
	W[4,4] = 0.0001
	W[5,5] = 0.0001
	
	for j in range(0, N): # downsample the number of observations
		isVisited.append(observedIndices[j])
		zt = observations[:, observedIndices[j]]
		coord = stereoModel(zt, fsu, fsv, cu, cv, b) # optical coordinates
		
		robo_coord = imu_T_cam @ coord # rot @ coord
		prev_mew_landmarks[3*observedIndices[j]:3*observedIndices[j]+3] = robo_coord[:3]

	for i in range(1, len(t[0])):
		print("t = ", i)
		# predict step
		tau = t[0][i] - prev_t
		v = linear_velocity[:, i]
		w = angular_velocity[:, i]
		u = np.hstack((v, w))
		u_hat = uHat(u)

		next_mew_robot = np.matmul(prev_mew_robot, scipy.linalg.expm(tau*u_hat))

		# do stuff with mew here:
		robot_cov = prev_cov[3*M:3*M+6,3*M:3*M+6]
		cov = (scipy.linalg.expm(-1*tau*uCurly(u)) @ robot_cov) @ scipy.linalg.expm(-1*tau*uCurly(u)).T + W
		
		cov_pred = np.identity(3*M + 6)
		cov_pred[3*M:3*M+6,3*M:3*M+6] = cov # should be 6x6

		# assign landmark to be the same
		cov_pred[0:3*M, 0:3*M] = prev_cov[0:3*M, 0:3*M]

		pred_mew_robot = next_mew_robot
		prev_t = t[0][i]

		# additional cross correlation values (top right)
		# LR predict
		LR = prev_cov[0:3*M, 3*M:3*M+6]
		covLR = LR@scipy.linalg.expm(-1*tau*uCurly(u)).T
		cov_pred[0:3*M, 3*M:3*M+6] = covLR

		# RL predict (bottom left)
		RL = prev_cov[3*M:3*M+6, 0:3*M]
		covRL = scipy.linalg.expm(-1*tau*uCurly(u))@RL
		cov_pred[3*M:3*M+6, 0:3*M] = covRL

		# check for landmarks visited for the first time
		observations = strided_features[:, :, i] # valid observations

		observedIndices = np.where(np.sum(observations, axis=0) != -4)[0]
		# remove observations observed for the first time and initialize them
		toDelete = []
		for j in range(len(observedIndices)):
			if observedIndices[j] not in isVisited:
				zt = observations[:, observedIndices[j]]
				coord = stereoModel(zt, fsu, fsv, cu, cv, b)
				robo_coord = pred_mew_robot @ imu_T_cam @ coord
				prev_mew_landmarks[3*observedIndices[j]:3*observedIndices[j]+3] = robo_coord[:3]
				isVisited.append(observedIndices[j])
				toDelete.append(j)

		observedIndices = np.delete(observedIndices, toDelete)
		N = len(observedIndices)
		I = np.identity(N)

		# EKF prep for ztilda and H
		ztildes_robot = np.zeros(4*N)
		zs = np.zeros(4*N)
		Hs_landmark = np.zeros((4*N, 3*M))
		Hs_robot = np.zeros((4*N, 6))

		for j in range(N):
			# mutual for both landmarks and robot
			zs[4*j:4*j+4] = observations[:, observedIndices[j]]

			# compute H for landmarks
			arg = (O_T_I @ np.linalg.inv(pred_mew_robot)) @ homo(prev_mew_landmarks[3*observedIndices[j]:3*observedIndices[j]+3])
			H = (((Ks @ (projDeriv(arg))) @ O_T_I) @ np.linalg.inv(pred_mew_robot)) @ P.T
			Hs_landmark[4*j:4*j+4,3*observedIndices[j]:3*observedIndices[j]+3] = H

			# compute ztilde for robot
			arg = (O_T_I @ np.linalg.inv(pred_mew_robot)) @ homo(prev_mew_landmarks[3*observedIndices[j]:3*observedIndices[j]+3])
			ztilde = Ks @ proj(arg)
			ztildes_robot[4*j:4*j+4] = ztilde

			# compute H for robot 
			arg = (O_T_I @ np.linalg.inv(pred_mew_robot)) @ homo(prev_mew_landmarks[3*observedIndices[j]:3*observedIndices[j]+3])
			H = (((-1*Ks) @ projDeriv(arg)) @ O_T_I) @ circleDot(np.linalg.inv(pred_mew_robot) @ homo(prev_mew_landmarks[3*observedIndices[j]:3*observedIndices[j]+3]))
			
			Hs_robot[4*j: 4*j+4, 0:6] = H

		# combine Hs_landmark and Hs_robot
		H_slam = np.hstack((Hs_landmark, Hs_robot))

		# EKF update. Separate update for each of the means
		K_next = (cov_pred @ H_slam.T) @ (np.linalg.inv((H_slam @ cov_pred) @ H_slam.T + np.kron(I, V)))
		K_landmarks = K_next[0:3*M, 0:4*N]
		K_robot = K_next[3*M:3*M+6, 0:4*N]
		next_landmark_mew = prev_mew_landmarks + K_landmarks @ (zs - ztildes_robot)
		next_robot_mew = pred_mew_robot @ scipy.linalg.expm(uHat(K_robot @ (zs - ztildes_robot)))

		I_big = np.identity(len((K_next @ H_slam)[0]))
		cov_next = (I_big - K_next @ H_slam) @ cov_pred

		prev_mew_landmarks = next_landmark_mew
		prev_mew_robot = next_robot_mew
		poses.append(prev_mew_robot)
		prev_cov = cov_next

	np_means = np.array(poses)
	reshaped_means = np.transpose(np_means, [1,2,0])
	visualize_trajectory_2d2(reshaped_means, prev_mew_landmarks)
	
	# You can use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)
