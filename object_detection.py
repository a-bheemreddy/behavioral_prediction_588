import numpy as np
from mmdet3d.apis import init_model, inference_detector
# from mmdet3d.datasets import LiDARDataset

import numpy as np
# from mmdet3d.core.bbox import LiDARInstance3DBoxes

def convert_npy_to_kitti(npy_file):
    # Load the LiDAR points from the .npy file
    points = np.load(npy_file)
    
    # Create a dictionary to hold the point cloud data
    data = {
        'type': 'LiDAR',
        'velodyne_path': 'placeholder_path',  # Placeholder path, since the data is loaded from .npy file
        'velodyne_points': points,
        'sample_idx': 0,  # Dummy sample index
        # 'boxes_3d_lidar': LiDARInstance3DBoxes(np.array([[0, 0, 0, 1, 1, 1, 0]])),  # Dummy boxes, replace if needed
        # 'gt_names': np.array(['Car']),  # Dummy class names, replace if needed
    }
    
    return data

# Example usage
npy_file = 'lidar1.npz'
kitti_data = convert_npy_to_kitti(npy_file)
# Step 2: Initialize the pretrained model

# Make sure to run: "git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x", otherwise
# the path to the config_file will not be valid (or exist).
config_file = 'mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
checkpoint_file = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # Use 'cpu' if GPU is not available

# Step 3: Perform inference on your custom LiDAR data
result, data = inference_detector(model, kitti_data)

# Process detection results
for pred_dict in result:
    # Print the predicted bounding boxes and scores
    print(pred_dict['boxes_3d'])
    print(pred_dict['scores_3d'])