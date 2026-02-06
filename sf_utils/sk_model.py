import numpy as np
import nimblephysics as nimble
import os
import pandas as pd

from COMETH import Skeleton, DynamicSkeleton

def create_body_model(action_name: str, vicon_path: str = './totalcapture/vicon', body_node_list: list = ['humerus_r','humerus_l',"thorax"], comp_model:bool = False):
    """ 
    This function initializes the DynamicSkeleton body model with the vicon data of the given subject
    Parameters:
        - action_name: The name of the specific action to consider ('subj/action')
        - action_dict: The dict of the specific action to consider
    Returns:
        The DynamicSkeleton
    """

    subj = action_name.split('/')[-2]
    action = action_name.split('/')[-1]
    path = os.path.join(vicon_path,action_name,'vicon_'+subj+'_'+action+'.csv')

    # Build skeleton
    s12 = Skeleton('BODY12.xml')
    # BSM
    s = DynamicSkeleton(config='BODY15_constrained_3D.xml',osim_file=os.path.abspath('COMETH/bsm_upper.osim'))
    s.hip_correction = False

    # Read data from CSV
    markers = pd.read_csv(path)
    # Build the markers dataframe with only the subset we are interested in
    Rz = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    Rx = np.array([
        [1,0,0 ],
        [0,0,-1 ],
        [0,1,0]
    ])

    markers.columns
    markers_dict = {
        'RKnee': 'right_knee',
        'LWrist': 'left_wrist',
        'RHip': 'right_hip',
        'RShoulder': 'right_shoulder',
        'LElbow': 'left_elbow',
        'LHip': 'left_hip',
        'RElbow': 'right_elbow',
        'RWrist': 'right_wrist',
        'LKnee': 'left_knee',
        'LShoulder': 'left_shoulder',
        'RAnkle': 'right_ankle',
        'LAnkle': 'left_ankle'
    }

    if not comp_model:
    # Offset deducted from TotalCapture videos
        offset_dict = {
            'ulna_r': [0,0,0],
            'ulna_l': [0,0,0],
            'humerus_r': [-0.04,-0.15,0],
            'humerus_l': [-0.04,-0.15,0],
            'thorax': [0.1, -0.1, 0.05]
            #TODO: complete with other possible accelerometer
        }

        rotation_dict = {
            'humerus_r': np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
            'humerus_l': np.array([[0,0,-1],[1,0,0],[0,-1,0]]),
            'thorax': np.array([[0,0,1],[1,0,0],[0,1,0]])
            #TODO: complete with other possible accelerometer
        }
    else:
        offset_dict = {
            'ulna_r': [0.0,0,0],
            'ulna_l': [0.0,0,0],
            'humerus_r': [0.0,0,0],#[-0.04,-0.15,0],
            'humerus_l': [0.0,0,0],#[-0.04,-0.15,0],
            'thorax': [0.0,0,0],#[0.1, -0.1, 0.05]
            'hand_r': [0,0,0],
            'hand_l': [0,0,0],
            'camera': [0.1,-0.1,0.05]
        }

        rotation_dict = {
            'ulna_r': np.eye(3),
            'ulna_l': np.eye(3),
            'humerus_r': np.eye(3),
            'humerus_l': np.eye(3),
            'thorax': np.eye(3),
            'hand_r': np.eye(3),
            'hand_l': np.eye(3),
            'camera': np.eye(3)
        }


    # Calculating target position for first frame of the sequence
    row = []
    for kp in markers_dict.keys():
        p = np.array([markers[markers_dict[kp]+"_x"][0],markers[markers_dict[kp]+"_y"][0],markers[markers_dict[kp]+"_z"][0]])
        # rotate the 3d point -90 on the x axis (from y up to z up)
        p_n = Rz.dot(Rx.dot(p))
        row += p_n.tolist()
    target = np.array(row)

    # Move the body model using gt to the first position
    s.reset()
    s12.load_from_numpy(target[:].reshape(-1,3),s.kps)
    s.load_from_BODY12(s12)
    s.exact_scale()
    s._nimble.setGravity(np.array([0.0, 0.0, -9.81])) # NOTE: this value is good until it matches the same value inside euler_to_gravity

    
    from typing import List, Tuple

    # Creating sensors list and adding it to nimble model
    sensors: List[Tuple[nimble.dynamics.BodyNode, nimble.math.Isometry3]] = []
    for node in body_node_list:
        if node == 'camera':
            body_node: nimble.dynamics.BodyNode = s._nimble.getBodyNode('thorax')
        else:
            body_node: nimble.dynamics.BodyNode = s._nimble.getBodyNode(node)
        translation: np.ndarray = np.array(offset_dict[node])
        rotation: np.ndarray = np.array(rotation_dict[node])
        watch_offset: nimble.math.Isometry3 = nimble.math.Isometry3(rotation, translation)

        sensors.append((body_node, watch_offset))
    
    s.IMUs = sensors

    return s

def compute_gt(action_name: str, vicon_path: str, comp_model: bool = False):
    subj = action_name.split('/')[-2]
    action = action_name.split('/')[-1]
    path = os.path.join(vicon_path,action_name,'vicon_'+subj+'_'+action+'.csv')

    # Build skeleton
    s12 = Skeleton('BODY12.xml')
    # BSM
    s = DynamicSkeleton(config='BODY15_constrained_3D.xml',osim_file=os.path.abspath('COMETH/bsm_upper.osim'))
    s.hip_correction = False

    # Read data from CSV
    markers = pd.read_csv(path)

    # Build the markers dataframe with only the subset we are interested in
    Rz = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    Rx = np.array([
        [1,0,0 ],
        [0,0,-1 ],
        [0,1,0]
    ])

    markers.columns
    markers_list = ['right_knee','left_wrist','right_hip','right_shoulder','left_elbow','left_hip','right_elbow','right_wrist','left_knee','left_shoulder','right_ankle','left_ankle']
        
    target = []
    for i in range(markers.shape[0]):
        row = []
        for kp in markers_list:
            p = np.array([markers[kp+"_x"][i],markers[kp+"_y"][i],markers[kp+"_z"][i]])
            # rotate the 3d point -90 on the x axis (from y up to z up)
            p_n = Rz.dot(Rx.dot(p))
            # print(p,p_n)
            row += p_n.tolist()
        target.append(np.array(row).reshape(-1,3))
    target = np.array(target)
    # Move the body model using gt to the first position
    s.reset()
    s12.load_from_numpy(target[0,:].reshape(-1,3),s.kps)
    s.load_from_BODY12(s12)
    s.exact_scale()
    
    q0 = s._nimble.getPositions().copy()

    gt = []
    offset = []
    # For each frame, calculate GT
    if not comp_model:
        for i in range(0, target.shape[0]):
            #Update gt position
            s12.load_from_numpy(target[i,:].reshape(-1,3),s.kps)
            s.load_from_BODY12(s12)
            s.exact_scale(max_iterations=1000, to_scale=False)
            q_gt = s._nimble.getPositions().copy()
            offset.append((q_gt[3:6] - q0[3:6]).reshape(1,3))
            q_gt[3:6] = q0[3:6] # Lock pelvis translation
            gt.append(q_gt.reshape(-1,1))
    else:
        for i in range(0, target.shape[0]):
            #Update gt position
            s12.load_from_numpy(target[i,:].reshape(-1,3),s.kps)
            s.load_from_BODY12(s12)
            s.exact_scale(max_iterations=1000, to_scale=False)
            q_thr = s._nimble.getBodyNode("thorax").getWorldTransform().matrix()[:3,3].reshape(3).copy()
            offset.append((q_thr).reshape(1,3))
            q_gt = s._nimble.getPositions().copy()
            q_gt[3:6] = q0[3:6] 
            gt.append(q_gt.reshape(-1,1))
        
    return target, np.array(gt), np.array(offset)