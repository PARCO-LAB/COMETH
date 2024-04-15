from Skeleton.Skeleton import Skeleton,ConstrainedSkeleton
import nimblephysics as nimble
import torch
import numpy as np
import os

class KinematicSkeleton(ConstrainedSkeleton):
    def __init__(self, config, name=None, osim_file=None, geometry_dir=''):
        
        super().__init__(config, name)
        
        self.keypoints_dict = {obj.name: obj for obj in self.keypoints_list}

        if osim_file is not None:
            # rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
            rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(osim_file,geometry_dir)
            self.type = 'BSM'
        else:
            # print(osim_file)
            # rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(osim_file)
            rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
            self.type = 'rajagopal'
            # rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim('bsm.osim')
        self._nimble: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton
        
        RASI = np.array([0,0.005,0.13])
        LASI = np.array([0,0.005,-0.13])
        LPSI = np.array([-0.14,0.015,-0.07])
        RPSI = np.array([-0.14,0.015,+0.07])
        
        RCAJ = np.array([0.015,-0.035,-0.02])
        RHGT = np.array([-0.05,0,0])
        LCAJ = np.array([0.015,-0.035,0.02])
        LHGT = np.array([-0.05,0,0])
        
        self.RShoulder = (RCAJ+RHGT)/2
        self.LShoulder = (LCAJ+LHGT)/2
        
        self.RHip = (RASI+RPSI)/2
        self.LHip = (LASI+LPSI)/2        
        
        self.s12_base = Skeleton('BODY12.xml')
        self.skeleton_from_nimble = ConstrainedSkeleton('BODY15_constrained_3D.xml')
        
        if  self.type == 'rajagopal':
            self.kps =  ['RKnee', 'LWrist', 'RHip', 'RShoulder',  'LElbow', 'LHip', 'RElbow', 'RWrist', 'LKnee', 'LShoulder', 'RAnkle', 'LAnkle']
            nimble_joint_names = [ 'walker_knee_r', 'radius_hand_l', 'hip_r', 'acromial_r', 'elbow_l', 'hip_l', 'elbow_r', 'radius_hand_r',  \
                        'walker_knee_l', 'acromial_l', 'ankle_r', 'ankle_l']
            self.body_dict = {'pelvis' : 'LPelvis',#LPelvis
                        'femur_r' : 'RFemur',
                        'tibia_r' : 'RTibia',
                        'talus_r' : '',
                        'calcn_r' : '',
                        'toes_r' : '',
                        'femur_l' : 'LFemur',
                        'tibia_l' : 'LTibia',
                        'talus_l' : '',
                        'calcn_l' : '',
                        'toes_l' : '',
                        'torso' : 'LClavicle',
                        'humerus_r' : 'RHumerus',
                        'ulna_r' : 'RHumerus',
                        'radius_r' : 'RForearm',
                        'hand_r' : '',
                        'humerus_l' : 'LHumerus',
                        'ulna_l' : 'LHumerus',
                        'radius_l' : 'LForearm',
                        'hand_l' : ''}
        elif self.type == "BSM":
            self.kps =  ['RKnee', 'LWrist', 'RHip', 'RShoulder',  'LElbow', 'LHip', 'RElbow', 'RWrist', 'LKnee', 'LShoulder', 'RAnkle', 'LAnkle']
            nimble_joint_names = [ 'walker_knee_r', 'wrist_l', 'hip_r', 'GlenoHumeral_r', 'elbow_l', 'hip_l', 'elbow_r', 'wrist_r',  \
                        'walker_knee_l', 'GlenoHumeral_l', 'ankle_r', 'ankle_l']
            self.body_dict = {  'pelvis':'Core', #LPelvis
                                'femur_r':'RFemur',
                                'tibia_r':'RTibia',
                                'talus_r':'',
                                'calcn_r':'',
                                'toes_r':'',
                                'femur_l':'LFemur',
                                'tibia_l':'LTibia',
                                'talus_l':'',
                                'calcn_l':'',
                                'toes_l':'',
                                'lumbar_body':'Core',#LClavicle
                                'thorax':'Core',#LClavicle
                                'head':'',
                                'scapula_r':'Core',#LClavicle
                                'humerus_r':'RHumerus',
                                'ulna_r':'RForearm',
                                'radius_r':'RForearm',
                                'hand_r':'',
                                'scapula_l':'Core',#LClavicle
                                'humerus_l':'LHumerus',
                                'ulna_l':'LForearm',
                                'radius_l':'LForearm',
                                'hand_l':''}

        self.joints = [self._nimble.getJoint(l) for l in nimble_joint_names]
        pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        self.s12_base.load_from_numpy(pos.reshape(-1,3),self.kps)
        self.skeleton_from_nimble.load_from_BODY12(self.s12_base)
        
        
    # Remove the joint position to place the corrected hip from marker-based
    def correct(self,pos):
        # Correct the pelvis joint
        transform = self._nimble.getBodyNode('pelvis').getWorldTransform()
        pos[3*self.kps.index("RHip"):3*self.kps.index("RHip")+3] = transform.multiply(self.RHip)
        pos[3*self.kps.index("LHip"):3*self.kps.index("LHip")+3] = transform.multiply(self.LHip)
        
        # # Correct the scapula joints
        transform = self._nimble.getBodyNode('scapula_l').getWorldTransform()
        pos[3*self.kps.index("LShoulder"):3*self.kps.index("LShoulder")+3] = transform.multiply(self.LShoulder)
        transform = self._nimble.getBodyNode('scapula_r').getWorldTransform()
        pos[3*self.kps.index("RShoulder"):3*self.kps.index("RShoulder")+3] = transform.multiply(self.RShoulder)
        return pos
    
    def scale(self):
        scale =  self._nimble.getBodyScales().reshape(-1,3)
        for i,b in enumerate(self.body_dict.keys()):
            if self.body_dict[b] == 'Core':
                scale[i,:] = self.estimate_height() / self.skeleton_from_nimble.estimate_height()
            elif self.body_dict[b] == '':
                scale[i,:] = self.estimate_height() / self.skeleton_from_nimble.estimate_height()
            else:
                scale[i,:] = self.bones_dict[self.body_dict[b]].length / self.skeleton_from_nimble.bones_dict[self.body_dict[b]].length # scale[i,:] * 
        self._nimble.setBodyScales(scale.reshape(-1,1))
    
    # Inverse kinematics through gradient descend
    def fit(self,max_iterations=100):
        target = super().to_numpy(self.kps).reshape(1,-1).squeeze()
        q = self._nimble.getPositions()
        i=0
        while i < max_iterations:
            pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
            d_loss_d__pos = 2 * (pos - target)
            d_pos_d_joint_angles = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
            d_loss_d_joint_angles = d_pos_d_joint_angles.T @ d_loss_d__pos
            
            # Lock the scapula
            # print(d_loss_d_joint_angles.shape)
            d_loss_d_joint_angles[29:32] = 0
            d_loss_d_joint_angles[39:42] = 0
            d_loss_d_joint_angles[20:22] = 0
            d_loss_d_joint_angles[23:25] = 0
            d_loss_d_joint_angles[26:29] = 0
            
            q -= 0.05 * d_loss_d_joint_angles
            self._nimble.setPositions(q)
            i+=1
    
    def IK():
        pass
    
    def to_numpy(self):
        return self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        