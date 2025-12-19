import numpy as np
import pandas as pd
import json
import os

#NOTE: This class is designed to navigate inside TotalCapture directory structure and files
class ImuData:
    def __init__(self, path: str):
        """ 
        Parameters
         path: str root directory of the imu sequence
        """
        self.file_path = path
        self.action = path.split('_')[-1].split('.')[0]
        self.subj = path.split('_')[-2]
        self.root_path, self.file_name = os.path.split(path)
        self.cal_file = self.subj + '_' + self.action + '_calibration.json'
        self.acc_read = None
        
    def read_imu_csv(self):
        """ 
        This function return a dict of numpy matrices (n_t x n_comp) representing the imu data of the given .csv.
        """
        # Loading data from files
        data = pd.read_csv(self.file_path, index_col=0)
        if data is None:
            return

        with open(os.path.join(self.root_path, self.cal_file), 'r') as file:
            calibration = json.load(file)

        indexes_name = data.columns.values

        # Obtaining acc names
        acc_readings = {}
        for idx in indexes_name[list(range(2,len(indexes_name),16))]:
            if 'left' in idx or 'right' in idx:
                name = idx.split('_')[0] +'_'+ idx.split('_')[1]
            else:
                name = idx.split('_')[0]
            acc_readings[name] = {}

            quaternion = data.loc[:,[name+'_quat_w', name+'_quat_x', name+'_quat_y', name+'_quat_z']]
            acc_readings[name]['quater']= quaternion.to_numpy()

            accel = data.loc[:,[name+'_acc_x', name+'_acc_y', name+'_acc_z']]
            acc_readings[name]['accel']= accel.to_numpy()

            gyro = data.loc[:,[name+'_gyro_x', name+'_gyro_y', name+'_gyro_z']]
            acc_readings[name]['gyro']= np.deg2rad(gyro.to_numpy())

            comp = data.loc[:,[name+'_comp_x', name+'_comp_y', name+'_comp_z']]
            acc_readings[name]['comp']= comp.to_numpy()

            grav = data.loc[:,[name+'_g_x', name+'_g_y', name+'_g_z']]
            acc_readings[name]['grav']= grav.to_numpy()

            acc_readings[name]['calib_bone'] = np.array(calibration['bone'][name])
            acc_readings[name]['calib_ref'] = np.array(calibration['ref'][name])
        
        acc_readings['n_frames'] = len(data.index)

        self.acc_read = acc_readings

    def get_imu_array(self, joint_list):
        acc =[]
        gyro = []
        for joint in joint_list:
            acc.append(self.acc_read[joint]['accel'])
            gyro.append(self.acc_read[joint]['gyro'])

        return np.array(acc), np.array(gyro)
    
    def get_world_orient(self, joint_list):
        ori = []
        for joint in joint_list:
            ori.append(self.acc_read[joint]['quater'])
        
        return np.array(ori)
    

#TODO: more testing
def camera_simulation(body_pos, camera_pos, fov_h = 1.78024, fov_v = 0.994838, R_cam = np.eye(3)):
    """
    Function use to simulate an egocentric camera behavior in identifing human keypoint. Camera considerata con lo z in fuori.
    ## Args
    - body_pos : human body keypoints 3d absolute positions as a matrix (N,3)
    - camera_pos : camera absolute 3D position as (3)
    - fov_h, fov_v : angular width of camera view in rad (horizontal and vertical)
    - R_cam : orientation of camera as a (3,3) Rotation matrix (camera --> world)
    ## Return
    - The body_pos array, with NaN if the joint is not visible
    """
    points_rel = body_pos - camera_pos # Nx3
    points_cam = points_rel @ R_cam  # Nx3

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    in_front = z > 0

    theta_h = np.arctan2(x, z)
    theta_v = np.arctan2(y, z)

    in_fov_h = np.abs(theta_h) <= fov_h / 2
    in_fov_v = np.abs(theta_v) <= fov_v / 2

    visible = in_front & in_fov_h & in_fov_v
    
    if not np.any(visible):
        return None

    # Nan for non-visible joints
    points_filtered = body_pos.copy().astype(float)
    points_filtered[~visible] = np.nan

    return points_filtered