from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np

# Initialize Kalman filters for each pedestrian
kalman_filters = {}

def update_pedestrian_tracking(bounding_boxes):
    # Predict the next state for each pedestrian
    predicted_states = {}
    for pedestrian_id, kalman_filter in kalman_filters.items():
        kalman_filter.predict()
        predicted_states[pedestrian_id] = kalman_filter.x[:3]  # Extract position
    
    # Perform data association
    cost_matrix = compute_cost_matrix(predicted_states, bounding_boxes)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Update matched pedestrians
    for row_idx, col_idx in zip(row_indices, col_indices):
        pedestrian_id = list(predicted_states.keys())[row_idx]
        kalman_filter = kalman_filters[pedestrian_id]
        kalman_filter.update(bounding_boxes[col_idx])
    
    # Create new tracks for unmatched bounding boxes
    for col_idx in set(range(len(bounding_boxes))) - set(col_indices):
        pedestrian_id = generate_new_pedestrian_id()
        kalman_filter = create_kalman_filter(bounding_boxes[col_idx])
        kalman_filters[pedestrian_id] = kalman_filter
    
    # Delete old tracks that are not updated for a certain time
    delete_old_tracks()
    
    # Return the tracked pedestrians
    tracked_pedestrians = {
        pedestrian_id: kalman_filter.x[:3] for pedestrian_id, kalman_filter in kalman_filters.items()
    }
    return tracked_pedestrians
        
def compute_cost_matrix(predicted_states, bounding_boxes):
    cost_matrix = np.zeros((len(predicted_states), len(bounding_boxes)))
    for i, (_, predicted_state) in enumerate(predicted_states.items()):
        for j, bounding_box in enumerate(bounding_boxes):
            cost_matrix[i, j] = np.linalg.norm(predicted_state - bounding_box[:3])
    return cost_matrix

def create_kalman_filter(bounding_box):
    kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
    
    # State transition matrix (assuming constant velocity model)
    kalman_filter.F = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    
    # Measurement matrix (assuming we only observe position)
    kalman_filter.H = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]])
    
    # Covariance matrices
    kalman_filter.P *= 1000  # Initial state covariance
    kalman_filter.Q = np.eye(6) * 0.01  # Process noise covariance
    kalman_filter.R = np.eye(3) * 1  # Measurement noise covariance
    
    # Initialize the state with the bounding box coordinates
    kalman_filter.x[:3] = bounding_box[:3]
    
    return kalman_filter

def generate_new_pedestrian_id():
    return str(len(kalman_filters) + 1)

def delete_old_tracks():
    to_delete = []
    for pedestrian_id, kalman_filter in kalman_filters.items():
        if kalman_filter.time_since_update > max_age:
            to_delete.append(pedestrian_id)
    for pedestrian_id in to_delete:
        del kalman_filters[pedestrian_id]