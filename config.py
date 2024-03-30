import numpy as np

'''
CONFIG FILE for Kalman filter

MUST PROVIDE:
1: dim_x = the dimensions for state)

2: dim_z = the dimensions for measurement

3: F = state transition matrix. Must be shape (dim_x, dim_x)

4: H = measurement matrix. Must be shape (dim_z, dim_x)

5: P = Covariance matrix for initial observation. Represents uncertainty of initial state.
    Must be matrix of shape (dim_x, dim_x)
    
6: Q = Process Covariance Matrix. Represents uncertainty of process
    Must be matrix of shape (dim_x, dim_x)
    
7: R = Measurement Covariance Matrix. Represents uncertainty of measurement
    Must be matrix of shape (dim_z, dim_z)
    
8: max_age = After how many time_steps with no observation should we delete a kalman tracker.
    Must be positive int
    
9: cost_function(predicted_state, observation) =
    function which calculates cost (or dissimilarity) between a
    predicted state (size dim_x) and an observation (size dim_z)

10: threshold = max limit for whether we consider a predicted state and new observation a match

11: initial_state(measurement) = 
     function which provides initial state, given just the first observation
     measurement is vector of size (dim_z). Output is vector of size (dim_x)
      

OPTIONAL:

'''

# state is vector with [x,y,vel_x, vel_y]
dim_x = 4
# measurements (sensor observations/bounding boxes) are just [x,y]
dim_z = 2

F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

P = np.eye(dim_x) * 100

Q = np.eye(dim_x) * 0.01

R = np.eye(dim_z) * 1

max_age = 6

threshold = 10

# distance between the predicted position (predicted_state[0:2]) and observed measurement
def cost_function(predicted_state, measurement):
    return np.linalg.norm(predicted_state[0:2] - measurement)

# given measurement of [x,y], set state to [x,y,0,0]
def initial_state(measurement):
    state = np.zeros(dim_x)
    state[0:dim_z] = measurement
    return state