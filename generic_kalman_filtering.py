from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
import importlib.util

# Specify the path to the config file
config_file_path = 'config.py'

# Load and import the config file
spec = importlib.util.spec_from_file_location("config", config_file_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

# Initialize Kalman filters for each pedestrian
# '##' is special key, which stores highest ID of a pedestrain
kalman_filters = {}
max_id = 0

def update_pedestrian_tracking(bounding_boxes):
    
    # Predict the next state for each pedestrian using past state
    predicted_states = {}
    for pedestrian_id, kalman_filter in kalman_filters.items():
        kalman_filter.predict()
        # kalman_filter.x now stores the prediction for the future state.
        predicted_states[pedestrian_id] = kalman_filter.x
    
    # Match observed bounding boxes with predicted future states
    cost_matrix, pedestrian_id_list = compute_cost_matrix(predicted_states, bounding_boxes)
    matches = compute_matching(cost_matrix, pedestrian_id_list, config.threshold)
    
    # Update matched pedestrians
    matched_pedestrians = set()
    matched_bboxes = set()
    for pedestrian_id, bbox_idx in matches.items():
        kalman_filters[pedestrian_id].update(bounding_boxes[bbox_idx])
        kalman_filters[pedestrian_id].time_since_update = 0
        matched_pedestrians.add(pedestrian_id)
        matched_bboxes.add(bbox_idx)
        
    # For unmatched Kalman filters, increase time since last update by 1
    for pedestrian_id in (set(kalman_filters.keys()) - matched_pedestrians):
        kalman_filters[pedestrian_id].time_since_update += 1
    delete_old_tracks(kalman_filters, config.max_age)
    
    # For unmatched bboxes, create a new kalman filter
    for col_idx in (set(range(len(bounding_boxes))) - matched_bboxes):
        pedestrian_id = generate_new_pedestrian_id()
        kalman_filters[pedestrian_id] = create_kalman_filter(bounding_boxes[col_idx])
    
    # Return the tracked pedestrians (mapping pedestrian ID to state)
    tracked_pedestrians = {
        pedestrian_id: kalman_filter.x for pedestrian_id, kalman_filter in kalman_filters.items()
    }
    return tracked_pedestrians



def compute_cost_matrix(predicted_states, bounding_boxes):
    cost_matrix = np.zeros((len(predicted_states), len(bounding_boxes)))
    ped_id_list = []
    for i, (ped_id, predicted_state) in enumerate(predicted_states.items()):
        ped_id_list.append(ped_id)
        for j, bounding_box in enumerate(bounding_boxes):
            cost_matrix[i, j] = config.cost_function(predicted_state, bounding_box)
    return cost_matrix, ped_id_list



def compute_matching(cost_matrix, ped_id_list, threshold):
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Maps pedestrian ID to observed box index
    matching = {}
    for i in range(len(row_indices)):
        # row_indices[i], col_indices[i] is matched according to the linear solver
        # but, only include matches that have enough similarity (less than threshold)
        if cost_matrix[row_indices[i], col_indices[i]] <= threshold:
            matching[ped_id_list[i]] = col_indices[i]
    return matching



# Bounding box is a vector representing our measurements.
# Must be length config.dim_z
def create_kalman_filter(bounding_box):
    kalman_filter = KalmanFilter(dim_x=config.dim_x, dim_z=config.dim_z)
    
    # Initialize the state with the bounding box
    kalman_filter.x = config.initial_state(bounding_box)
    
    # State transition matrix
    kalman_filter.F = config.F
    
    # Measurement matrix
    kalman_filter.H = config.H
        
    # Initial state uncertainty
    kalman_filter.P = config.P
    
    # Process noise (uncertainty of process)
    kalman_filter.Q = config.Q
    
    # Measurement noise covariance (uncertainty of obtained measurements/bounding boxes)
    kalman_filter.R = config.R

    kalman_filter.time_since_update = 0
    
    return kalman_filter



def generate_new_pedestrian_id():
    max_id += 1
    return str(max_id)

def delete_old_tracks(kalman_filters, max_age):
    to_delete = []
    for pedestrian_id, kalman_filter in kalman_filters.items():
        if kalman_filter.time_since_update > max_age:
            to_delete.append(pedestrian_id)
    for pedestrian_id in to_delete:
        del kalman_filters[pedestrian_id]