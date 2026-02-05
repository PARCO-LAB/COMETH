import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from COMETH import Skeleton, DynamicSkeleton


class MetricManager:
    """ 
    Class to work with the metrics
    """
    def __init__(self, s, target, offset, calc: list = []):
        self.calculated = calc
        self.s = s
        self.target= target
        self.offset = offset
        self.joint_arm_dict = {
            0:'shoulder_right',
            1:'elbow_right',
            2:'wrist_right',
            3:'shoulder_left',
            4:'elbow_left',
            5:'wrist_left',
        }
        self.joint_pos_dict = {
            0:'walker_knee_r', 
            1:'wrist_l',
            2:'hip_r',
            3:'GlenoHumeral_r',
            4:'elbow_l',
            5:'hip_l', 
            6:'elbow_r', 
            7:'wrist_r',
            8:'walker_knee_l',
            9:'GlenoHumeral_l',
            10:'ankle_r',
            11:'ankle_l'
        }
    
    def append_new_pos(self, pos):
        self.calculated.append(pos.reshape(-1,1))
    
    def __mpjpe(self):
        """ 
        Mean per joint positional error (over only interest joints)
        """
        #wrist + glenohumeral + elbow
        #wrist_l + glenohumeral_r + elbow_l +elbow_r + wrist_r + GlenoHumeral_l'

        pos = self.s._nimble.getPositions()

        calc_pos = []
        n_frames = len(self.calculated)
        for idx in range(n_frames):
            self.s._nimble.setPositions(self.calculated[idx])
            calc_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))

        self.s._nimble.setPositions(pos)
        
        calc_pos = np.array(calc_pos).reshape(n_frames,-1,3)

        if not len(self.target)==len(calc):
            gt = self.target[:len(calc)]
            offset = self.offset[:len(calc)]
        else:
            gt = self.target
            offset = self.offset


        calc_pos = calc_pos + offset
        mpjpe = np.linalg.norm(calc_pos[:,[3,6,7,9,4,1],:]-gt[:,[3,6,7,9,4,1],:], axis=-1)
        calc = calc.reshape(-1,36)
        gt = gt.reshape(-1,36)

        return mpjpe, calc_pos, gt, offset
     
    def save_metrics(self, metrics_list: list = ['mpjpe'], dir: str = './metrics'):

        if not os.path.exists(dir):
            os.mkdir(dir)
        
        for metric in metrics_list:
            if metric == 'mpjpe':
                mpjpe, calc_pos, gt_pos, offset = self.__mpjpe()
                columns = []
                for i in range(6):
                    columns.append(self.joint_arm_dict[i])

                mpjpe_df = pd.DataFrame(mpjpe, columns=columns)
                mpjpe_df.to_csv(os.path.join(dir, 'mpjpe.csv'))


                columns = []
                for i in range(12):
                    columns.append(self.joint_pos_dict[i]+'_x')
                    columns.append(self.joint_pos_dict[i]+'_y')
                    columns.append(self.joint_pos_dict[i]+'_z')

                calc_df = pd.DataFrame(calc_pos, columns=columns)
                calc_df.to_csv(os.path.join(dir, 'calc_pos.csv'))

                offset_df = pd.DataFrame(offset.reshape(-1,3), columns=['x','y','z'])
                offset_df.to_csv(os.path.join(dir, 'offset.csv'))

                gt_df = pd.DataFrame(gt_pos, columns=columns)
                gt_df.to_csv(os.path.join(dir, 'gt_pos.csv'))

def save_metrics_comparative(seq_dir, pos_estimation, marker, offset):
    joint_arm_dict = {
        0:'shoulder_right',
        1:'elbow_right',
        2:'wrist_right',
        3:'shoulder_left',
        4:'elbow_left',
        5:'wrist_left',
    }

    joint_pos_dict = {
        0:'walker_knee_r', 
        1:'wrist_l',
        2:'hip_r',
        3:'shoulder_r',
        4:'elbow_l',
        5:'hip_l', 
        6:'elbow_r', 
        7:'wrist_r',
        8:'walker_knee_l',
        9:'shoulder_l',
        10:'ankle_r',
        11:'ankle_l'
    }


    
    calc = np.array(pos_estimation).reshape(-1,6,3).copy()
    if not len(marker)==len(calc):
        marker_new = marker[:len(calc)].copy()
        offset_new = offset[:len(calc)].copy()
    else:
        offset_new = offset.copy()
        marker_new = marker.copy()
    calc = calc + offset_new
    mpjpe = np.linalg.norm(calc-marker_new[:,[3,6,7,9,4,1],:], axis=-1)
    
    
    columns_gt = []
    for i in range(12):
        columns_gt.append(joint_pos_dict[i]+'_x')
        columns_gt.append(joint_pos_dict[i]+'_y')
        columns_gt.append(joint_pos_dict[i]+'_z')
    columns = []
    for i in range(6):
        columns.append(joint_arm_dict[i])
    columns_arm = []
    for i in range(6):
        columns_arm.append(joint_arm_dict[i]+'_x')
        columns_arm.append(joint_arm_dict[i]+'_y')
        columns_arm.append(joint_arm_dict[i]+'_z')

    mpjpe_df = pd.DataFrame(mpjpe, columns=columns)
    mpjpe_df.to_csv(os.path.join(seq_dir, 'mpjpe.csv'))

    calc = calc.reshape(-1,18)
    marker_new = marker_new.reshape(-1,36)

    offset_df = pd.DataFrame(offset_new.reshape(-1,3), columns=['x','y','z'])
    offset_df.to_csv(os.path.join(seq_dir, 'offset.csv'))

    calc_df = pd.DataFrame(calc, columns=columns_arm)
    calc_df.to_csv(os.path.join(seq_dir, 'pos_estimation.csv'))

    gt_df = pd.DataFrame(marker_new, columns=columns_gt)
    gt_df.to_csv(os.path.join(seq_dir, 'pos_gt.csv'))

