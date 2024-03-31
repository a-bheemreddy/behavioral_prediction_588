import numpy as np
from generic_kalman_filtering_class import KalmanTracker
import sys

def read_ground_truth_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            time_step = int(values[0])
            ped_id = int(values[1])
            x = float(values[2])
            y = float(values[3])
            data.append((time_step, ped_id, x, y))
    return data

def convert_txt_to_csv(file_path):
    with open(file_path, 'r') as txt_file:
        with open('ground_truth.csv', 'w') as csv_file:
            csv_file.write('time_step,ped_id,x,y\n')
            for line in txt_file:
                csv_file.write(line.replace(' ', ','))

def add_noise(x, y, std_dev):
    noise_x = np.random.normal(0, std_dev)
    noise_y = np.random.normal(0, std_dev)
    return x + noise_x, y + noise_y

def test_kalman_tracker(ground_truth_data, noise_std_dev):
    tracker = KalmanTracker('config.py')
    
    current_time_step = -1
    for time_step, ped_id, x, y in ground_truth_data:
        if time_step != current_time_step:
            # Update tracker with previous time step's measurements
            if current_time_step >= 0:
                tracked_pedestrians = tracker.update_pedestrian_tracking(measurements)
                print(f"Time step: {current_time_step}")
                print("Tracked pedestrians:", tracked_pedestrians)
                print()
            
            current_time_step = time_step
            measurements = []
        
        # Add noise to ground truth data
        noisy_x, noisy_y = add_noise(x, y, noise_std_dev)
        measurements.append(np.array([noisy_x, noisy_y]))
    
    # Update tracker with last time step's measurements
    tracked_pedestrians = tracker.update_pedestrian_tracking(measurements)
    print(f"Time step: {current_time_step}")
    print("Tracked pedestrians:", tracked_pedestrians)

if __name__ == '__main__':
    # Read ground truth data from file
    ground_truth_data = read_ground_truth_data('ground_truth.txt')

    # Set the standard deviation for the noise
    noise_std_dev = 0.5

    # Test the Kalman tracker
    test_kalman_tracker(ground_truth_data, noise_std_dev)