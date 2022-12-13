import numpy as np
# import scipy.stats
import time
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data, read_true_data, read_input_data
from matplotlib.patches import Ellipse

lxs = []
lys = []

#plot preferences, interactive plotting mode
# fig = plt.figure()
# plt.axis([-1, 12, 0, 10])
# plt.ion()
# plt.show()

def plot_state(mu, sigma, landmarks, map_limits):
    # global lxs
    # global lys
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        # original
        # lx.append(landmarks[i+1][0])
        # ly.append(landmarks[i+1][1])
        # mine
        lx.append(landmarks[i][0])
        ly.append(landmarks[i][1])

    # mean of belief as current estimate
    estimated_pose = mu

    #calculate and plot covariance ellipse
    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0],estimated_pose[1]], width=width, height=height, angle=angle/np.pi*180)
    ell.set_alpha(0.25)

    # lxs.append(estimated_pose[0])
    # lys.append(estimated_pose[1])

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)

def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 
    
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    #motion noise
    # Q = np.array([[0.2, 0.0, 0.0],\
    #             [0.0, 0.2, 0.0],\
    #             [0.0, 0.0, 0.02]])
    Q = np.array([[0.01, 0.0, 0.0],\
            [0.0, 0.01, 0.0],\
            [0.0, 0.0, 0.05]])

    #noise free motion
    x_new = x + delta_trans * np.cos(theta + delta_rot1)
    y_new = y + delta_trans * np.sin(theta + delta_rot1)
    theta_new = theta + delta_rot1 + delta_rot2
    #Jakobian of g with respect to the state
    G = np.array([[1.0, 0.0, -delta_trans * np.sin(theta + delta_rot1)],\
        [0.0, 1.0, delta_trans * np.cos(theta + delta_rot1)],\
        [0.0, 0.0, 1.0]])
    #new mu and sigma
    mu = [x_new, y_new, theta_new]
    sigma = np.dot(np.dot(G,sigma),np.transpose(G)) + Q
    
    return mu, sigma

def prediction_step_vel(input, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 
    
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    v = input['v']
    w = input['w']
    dt = 0.1 # input['dt']

    #motion noise
    # Q = np.array([[0.2, 0.0, 0.0],\
    #             [0.0, 0.2, 0.0],\
    #             [0.0, 0.0, 0.02]])
    Q = np.array([[0.01, 0.0, 0.0],\
            [0.0, 0.01, 0.0],\
            [0.0, 0.0, 0.05]])

    #noise free motion
    if w == 0:
        x_new = x + v*dt*np.cos(theta)
        y_new = y + v*dt*np.sin(theta)
    else:
        x_new = x + (-v/w)*np.sin(theta) + (v/w)*np.sin(theta + w*dt)
        y_new = y + (v/w)*np.cos(theta) - (v/w)*np.cos(theta + w*dt)
    theta_new = theta + w*dt
    #Jakobian of g with respect to the state
    if w == 0:
        G = np.array([[1.0, 0.0, -v*dt*np.sin(theta)],\
        [0.0, 1.0, v*dt*np.cos(theta)],\
        [0.0, 0.0, 1.0]])
    else:
        G = np.array([[1.0, 0.0, (v/w)*np.cos(theta) - (v/w)*np.cos(theta+w*dt)],\
            [0.0, 1.0, (v/w)*np.sin(theta) - (v/w)*np.sin(theta+w*dt)],\
            [0.0, 0.0, 1.0]])
    #new mu and sigma
    mu = [x_new, y_new, theta_new]
    sigma = np.dot(np.dot(G,sigma),np.transpose(G)) + Q
    
    return mu, sigma

def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    # Compute the expected range measurements for each landmark.
    # This corresponds to the function h
    H = []
    Z = []
    expected_ranges = []

    if len(ids) == 0:
        print("!")
        return mu, sigma
        
    for i in range(len(ids)):
        lm_id = ids[i]
        meas_range = ranges[i]
        lx = landmarks[lm_id][0]
        ly = landmarks[lm_id][1]

        #calculate expected range measurement
        range_exp = np.sqrt( (lx - x)**2 + (ly - y)**2 )

        # print(range_exp, meas_range, meas_range-range_exp)
        
        #compute a row of H for each measurement
        H_i = [(x - lx)/range_exp, (y - ly)/range_exp, 0]
        H.append(H_i)
        Z.append(meas_range)
        expected_ranges.append(range_exp)
    
    # print("sensor readings=", len(ids), ",  H=", len(H), len(H[0]))
    
    # noise covariance for the measurements
    # R = 0.5 * np.eye(len(ids))
    R = 0.1 * np.eye(len(ids))

    # Kalman gain
    K_help = np.linalg.inv(np.dot(np.dot(H,sigma),np.transpose(H)) + R) 
    K = np.dot(np.dot(sigma,np.transpose(H)),K_help)

    # Kalman correction of mean and covariance
    mu = mu + np.dot(K,(np.array(Z) - np.array(expected_ranges))) 
    sigma = np.dot(np.eye(len(sigma)) - np.dot(K,H),sigma)

    return mu, sigma

def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world_map1.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data_map1.dat")

    print("Reading input data")
    input_readings = read_input_data("../data/input_data_map1.dat")
    
    print("Reading true data")
    true_readings = read_true_data("../data/true_data_map1.dat")
    
    #initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],\
                      [0.0, 1.0, 0.0],\
                      [0.0, 0.0, 1.0]])

    # map_limits = [-1, 12, -1, 10]
    map_limits = [-2, 2, -2, 2]

    # save for plot
    mus = []
    odoms = []
    ts = []

    #run kalman filter
    for timestep in range(int(len(sensor_readings)/2)):
    # for timestep in range(len(sensor_readings)):

        #plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        #perform prediction step
        # mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma)
        # odoms.append(mu)

        #perform prediction step
        mu, sigma = prediction_step_vel(input_readings[timestep], mu, sigma)
        odoms.append(mu)


        #perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)
        mus.append(mu)

        ts.append(timestep)
        

    # print(lxs[:5])
    # print(lys[:5])
    # plt.scatter(lxs, lys)
    # plt.show()
    # print(len(mus), len(true_readings), len(odoms))
    
    # mus = np.array(mus)
    # true_readings = np.array(true_readings)
    # odoms = np.array(odoms)

    # plt.plot(true_readings[:, 0], true_readings[:, 1])
    # plt.plot(odoms[:, 0], odoms[:, 1])
    # plt.plot(mus[:, 0], mus[:, 1])
    # plt.legend('true', 'odom', 'ekf')

    # plt.plot(ts, true_readings[:, 0]-odoms[:, 0])
    # plt.plot(ts, true_readings[:, 0]-mus[:, 0])
    plt.show()

if __name__ == "__main__":
    main()