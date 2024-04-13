from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np

# Initialize Kalman filters for each pedestrian
# '##' is special key, which stores highest ID of pedestrain
kalman_filters = {'##':0}

def update_pedestrian_tracking(bounding_boxes):
    
    # Predict the next state for each pedestrian using past state
    predicted_states = {}
    for pedestrian_id, kalman_filter in kalman_filters.items():
        kalman_filter.predict()
        # kalman_filter.x now stores the prediction for the future state.
        # For macthing with observed bounding boxes, we require just the positions (x,y)
        predicted_states[pedestrian_id] = kalman_filter.x[:2]
    
    # Match observed bounding boxes with predicted future states
    bounding_boxes_pos = [bbox[0:2] for bbox in bounding_boxes]
    cost_matrix, pedestrian_id_list = compute_cost_matrix(predicted_states, bounding_boxes_pos)
    matches = compute_matching(cost_matrix, pedestrian_id_list, 5)
    
    # Update matched pedestrians
    matched_pedestrians = set()
    matched_bboxes = set()
    for pedestrian_id, bbox_idx in matches:
        kalman_filters[pedestrian_id].update(bounding_boxes[bbox_idx])
        matched_pedestrians.add(pedestrian_id)
        matched_bboxes.add(bbox_idx)
        
    # For unmatched Kalman filters, increase time since last update by 1
    for pedestrian_id in (set(kalman_filters.keys()) - matched_pedestrians):
        kalman_filters[pedestrian_id].time_since_update += 1
    delete_old_tracks(kalman_filters, 10)
    
    # For unmatched bboxes, create a new kalman filter
    for col_idx in (set(range(len(bounding_boxes))) - matched_bboxes):
        pedestrian_id = generate_new_pedestrian_id()
        kalman_filters[pedestrian_id] = create_kalman_filter(bounding_boxes[col_idx])
    
    # Return the tracked pedestrians (mapping pedestrian ID to state)
    tracked_pedestrians = {
        pedestrian_id: kalman_filter.x for pedestrian_id, kalman_filter in kalman_filters.items()
    }
    return tracked_pedestrians

def compute_cost_matrix(predicted_states, bounding_box_states):
    # Compute distance between center of boxes as similarity measurement
    cost_matrix = np.zeros((len(predicted_states), len(bounding_box_states)))

    ped_id_list = []
    for i, (ped_id, predicted_state) in enumerate(predicted_states.items()):
        ped_id_list.append(ped_id)
        for j, bounding_box in enumerate(bounding_box_states):
            cost_matrix[i, j] = np.linalg.norm(predicted_state - bounding_box)
    return cost_matrix, ped_id_list

def compute_matching(cost_matrix, ped_id_list, threshold):
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Maps pedestrian ID to observed box index
    matching = {}
    
    for i in range(len(row_indices)):
        # row_indices[i], col_indices[i] is matched
        # but, we will include matches that have enough similarity
        if cost_matrix[row_indices[i], col_indices[i]] <= threshold:
            matching[ped_id_list[i]] = col_indices[i]
        
    return matching

# bounding box is a 2D np.array 
# It is defined by (x,y). Assume x,y is center of pedestrian
def create_kalman_filter(bounding_box):
    
    # x is state with 6 variables: pos (x,y), box shape (l,w), velocity (vx, vy)
    kalman_filter = KalmanFilter(dim_x=6, dim_z=4)
    
    # Initialize the state (x,y,l,w,vx,vy) with the bounding box, and 0 velocity
    kalman_filter.x = np.zeros(6)
    kalman_filter.x[0:4] = bounding_box
    
    # State transition matrix (assuming constant velocity model)
    kalman_filter.F = np.array([[1, 0, 0, 0, 1, 0],
                                [0, 1, 0, 0, 0, 1],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
    
    # Measurement matrix (assuming we only observe position and box-size)
    kalman_filter.H = np.array([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0]])
    
    # Covariance matrices for uncertainty
    
    # Initial state uncertainty [P is already Identity]
    kalman_filter.P *= 100  
    
    # Initial state uncertainty for velocity will be higher, since it is set to 0 with no prior measurements
    kalman_filter.P[-2:, -2:] *= 10 
    
    # Process noise  (uncertainty of process)
    kalman_filter.Q = np.eye(6) * 0.01  
    
    # Measurement noise covariance (uncertainty of obtained bounding boxes)
    kalman_filter.R = np.eye(4) * 1
    
    kalman_filter.time_since_update = 0
    
    return kalman_filter

def generate_new_pedestrian_id(kalman_filters):
    kalman_filters['##'] += 1
    return str(kalman_filters['##'])

def delete_old_tracks(kalman_filters, max_age):
    to_delete = []
    for pedestrian_id, kalman_filter in kalman_filters.items():
        if kalman_filter.time_since_update > max_age:
            to_delete.append(pedestrian_id)
    for pedestrian_id in to_delete:
        del kalman_filters[pedestrian_id]