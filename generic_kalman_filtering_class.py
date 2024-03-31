from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
import importlib.util

class KalmanTracker:
    def __init__(self, config_file_path):
        spec = importlib.util.spec_from_file_location("config", config_file_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        self.kalman_filters = {}
        self.max_id = 0
               
        self.dim_x = config.dim_x
        self.dim_z = config.dim_z
        self.F = config.F
        self.H = config.H
        self.P = config.P
        self.Q = config.Q
        self.R = config.R
        self.max_age = config.max_age
        self.threshold = config.threshold
        self.cost_function = config.cost_function
        self.initial_state = config.initial_state
    
    # Must be called in loop to continuously update all tracked objects
    # bounding_boxes are just the list of measurements/observations/sensor readings
    def update_pedestrian_tracking(self, bounding_boxes):
    
        # Predict the next state for each pedestrian using past state
        predicted_states = {}
        for pedestrian_id, kalman_filter in self.kalman_filters.items():
            kalman_filter.predict()
            # kalman_filter.x now stores the prediction for the future state.
            predicted_states[pedestrian_id] = kalman_filter.x
        
        # Match observed bounding boxes with predicted future states
        cost_matrix, pedestrian_id_list = self.compute_cost_matrix(predicted_states, bounding_boxes)
        matches = self.compute_matching(cost_matrix, pedestrian_id_list)
        
        # Update matched pedestrians
        matched_pedestrians = set()
        matched_bboxes = set()
        for pedestrian_id, bbox_idx in matches.items():
            self.kalman_filters[pedestrian_id].update(bounding_boxes[bbox_idx])
            self.kalman_filters[pedestrian_id].time_since_update = 0
            matched_pedestrians.add(pedestrian_id)
            matched_bboxes.add(bbox_idx)
            
        # For unmatched Kalman filters, increase time since last update by 1
        for pedestrian_id in (set(self.kalman_filters.keys()) - matched_pedestrians):
            self.kalman_filters[pedestrian_id].time_since_update += 1
        self.delete_old_tracks()
        
        # For unmatched bboxes, create a new kalman filter
        for col_idx in (set(range(len(bounding_boxes))) - matched_bboxes):
            pedestrian_id = self.generate_new_pedestrian_id()
            self.kalman_filters[pedestrian_id] = self.create_kalman_filter(bounding_boxes[col_idx])
        
        # Return the tracked pedestrians (mapping pedestrian ID to state)
        tracked_pedestrians = {
            pedestrian_id: kalman_filter.x for pedestrian_id, kalman_filter in self.kalman_filters.items()
        }
        return tracked_pedestrians
    
    ### HELPER FUNCTIONS
    def generate_new_pedestrian_id(self):
        self.max_id += 1
        return str(self.max_id)

    def delete_old_tracks(self):
        to_delete = []
        for pedestrian_id, kalman_filter in self.kalman_filters.items():
            if kalman_filter.time_since_update > self.max_age:
                to_delete.append(pedestrian_id)
        for pedestrian_id in to_delete:
            del self.kalman_filters[pedestrian_id]
            
    def compute_cost_matrix(self,predicted_states, bounding_boxes):
        cost_matrix = np.zeros((len(predicted_states), len(bounding_boxes)))
        ped_id_list = []
        for i, (ped_id, predicted_state) in enumerate(predicted_states.items()):
            ped_id_list.append(ped_id)
            for j, bounding_box in enumerate(bounding_boxes):
                cost_matrix[i, j] = self.cost_function(predicted_state, bounding_box)
        return cost_matrix, ped_id_list
    
    def compute_matching(self, cost_matrix, ped_id_list):
        row_indices, col_indices = linear_sum_assignment(cost_matrix)        
        # Maps pedestrian ID to observed box index
        matching = {}
        for i in range(len(row_indices)):
            # row_indices[i], col_indices[i] is matched according to the linear solver
            # but, only include matches that have enough similarity (less than threshold)
            if cost_matrix[row_indices[i], col_indices[i]] <= self.threshold:
                matching[ped_id_list[i]] = col_indices[i]
        return matching
    
    def create_kalman_filter(self, bounding_box):
        kalman_filter = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        # Initialize the state with the bounding box
        kalman_filter.x = self.initial_state(bounding_box)
        
        # State transition matrix
        kalman_filter.F = self.F.copy()
        
        # Measurement matrix
        kalman_filter.H = self.H.copy()
            
        # Initial state uncertainty
        kalman_filter.P = self.P.copy()
        
        # Process noise (uncertainty of process)
        kalman_filter.Q = self.Q.copy()
        
        # Measurement noise covariance (uncertainty of obtained measurements/bounding boxes)
        kalman_filter.R = self.R.copy()
        
        kalman_filter.time_since_update = 0
        return kalman_filter
