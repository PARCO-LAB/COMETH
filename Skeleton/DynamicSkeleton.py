from Skeleton.Skeleton import Skeleton,ConstrainedSkeleton
import nimblephysics as nimble
import torch
import numpy as np
import os
import cvxpy as cp

class DynamicSkeleton(ConstrainedSkeleton):
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
        
        self.neutral_position = self._nimble.getPositions()
        
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

        self.q_l = np.ones((49))*(-180)
        self.q_u = np.ones((49))*180
        self.q_l[0:6] = -np.inf
        self.q_l[6] = -40
        self.q_l[7] = -45
        self.q_l[8] = -45
        self.q_l[13] =-40
        self.q_l[14] =-45
        self.q_l[15] =-45
        self.q_l[9] = -10
        self.q_l[16] =-10
        self.q_l[10] =-20
        self.q_l[17] =-20
        self.q_l[20] =-20
        self.q_l[23] =-20
        self.q_l[29] =-90
        self.q_l[39] =-90
        self.q_l[31] =-10
        self.q_l[41] =-10
        self.q_l[22] =-5
        self.q_l[25] =-5
        self.q_l[42] =0
        self.q_l[43] =-90
        self.q_l[44] =-60
        self.q_l[32] =-150
        self.q_l[33] =-70
        self.q_l[34] =-60
        self.q_l[45] =-6
        self.q_l[35] =-6
        self.q_l[30] = -8
        self.q_l[40] = -8
        self.q_l[21] = -5
        self.q_l[24] = -5
        # Upper limits
        self.q_u[0:6] = np.inf
        self.q_u[6]  = 140
        self.q_u[7]  = 45
        self.q_u[8]  = 45
        self.q_u[13] = 140
        self.q_u[31] = 40
        self.q_u[41] = 40
        self.q_u[14] = 45
        self.q_u[15] = 45
        self.q_u[9]  = 140
        self.q_u[16] = 140
        self.q_u[10] = 55
        self.q_u[17] = 55
        self.q_u[29] =-55
        self.q_u[39] =-55
        self.q_u[20] = 20
        self.q_u[23] = 20
        self.q_u[22] = 5
        self.q_u[25] = 5
        self.q_u[42] = 150
        self.q_u[43] = 70
        self.q_u[44] = 180
        self.q_u[32] = 0
        self.q_u[33] = 90
        self.q_u[34] = 180
        self.q_u[45] = 154
        self.q_u[35] = 154
        self.q_u[30] = 2
        self.q_u[40] = 2
        self.q_u[21] = 5
        self.q_u[24] = 5

        self.q_l = self.q_l*np.pi/180
        self.q_u = self.q_u*np.pi/180

        # self.qdot_l = np.array([-0.55,-0.43,-1.04,-0.74,-0.20,-0.30,-1.58,-0.57,-0.61,-1.97,0,0,0,-1.61,-0.56,-0.55,-1.97,0,0,0,-0.49,0,-0.31,-0.37,0,-0.29,0,0,0,0,0,0,-0.80,-0.84,-1.37,-1.34,-0.080,0,0,0,0,0,-0.80,-0.84,-1.42,-1.16,-0.070,0,0])
        # self.qdot_u = np.array([0.57,0.43,0.95,0.84,0.20,0.29,1.93,0.54,0.53,2.14,0,0,0,1.95,0.54,0.51,2.23,0,0,0,0.49,0,0.32,0.38,0,0.29,0,0,0,0,0,0,0.78,0.88,1.4,1.47,0.090,0,0,0,0,0,0.84,0.72,1.37,1.27,0.080,0,0])
        
        # Maximum Velocities in Flexion and Extension Actions for Sport
        self.qdot_l = np.zeros(self.q_u.shape)-10
        self.qdot_u = np.zeros(self.q_u.shape)+10
        
        self.prob = None

        self.joints = [self._nimble.getJoint(l) for l in nimble_joint_names]
        pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        self.s12_base.load_from_numpy(pos.reshape(-1,3),self.kps)
        self.skeleton_from_nimble.load_from_BODY12(self.s12_base)
        
    
    def reset_position(self):
        self._nimble.setPositions(self.neutral_position)
        
    
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
            if self.body_dict[b] == 'Core' or self.body_dict[b] == '':
                scale[i,:] = self.estimate_height() / self.skeleton_from_nimble.estimate_height()
            else:
                scale[i,:] = self.bones_dict[self.body_dict[b]].length / self.skeleton_from_nimble.bones_dict[self.body_dict[b]].length
        self._nimble.setBodyScales(scale.reshape(-1,1))
    
    # Inverse kinematics through gradient descend
    def gdIK(self,max_iterations=100):
        target = super().to_numpy(self.kps).reshape(1,-1).squeeze()
        q = self._nimble.getPositions()
        i=0
        older_loss = np.inf
        while i < max_iterations:
            pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
            
            error = pos - target
            loss = np.inner(error, error)
            if np.abs(older_loss - loss) < 0.00001:
                break
            older_loss = loss
            
            
            d_loss_d__pos = 2 * (pos - target)
            d_pos_d_joint_angles = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
            d_loss_d_joint_angles = d_pos_d_joint_angles.T @ d_loss_d__pos
            
            # Lock some joints
            # d_loss_d_joint_angles[29:32] = 0 # L scapula
            # d_loss_d_joint_angles[39:42] = 0 # R scapula
            # d_loss_d_joint_angles[21] = 0 # Lumbar extension
            # d_loss_d_joint_angles[24] = 0 # Thorax extension
            
            q -= 0.05 * d_loss_d_joint_angles
            
            # # Right Hip
            # q[6] = max(min(q[6], 140*np.pi/180), -40*np.pi/180)     # [-40,140]
            # q[7] = max(min(q[7], 45*np.pi/180), -45*np.pi/180)     # [-40,140]
            # q[8] = max(min(q[8], 45*np.pi/180), -45*np.pi/180)     # [-40,140]
            # # Left Hip
            # q[13] = max(min(q[13], 140*np.pi/180), -40*np.pi/180)     # [-40,140]
            # q[14] = max(min(q[14], 45*np.pi/180), -45*np.pi/180)     # [-40,140]
            # q[15] = max(min(q[15], 45*np.pi/180), -45*np.pi/180)     # [-40,140]
            # # Right Knee
            # q[9] = max(min(q[9], 140*np.pi/180), -10*np.pi/180)     # [-10,140]
            # # Left Knee
            # q[16] = max(min(q[16], 140*np.pi/180), -10*np.pi/180)     # [-10,140]
            # # Ankles
            # q[10] = max(min(q[10], 55*np.pi/180), -20*np.pi/180)     # [-10,140]
            # q[17] = max(min(q[17], 55*np.pi/180), -20*np.pi/180)     # [-10,140]
            # # # Lumbar amd thorax bending
            # q[20] = max(min(q[20], 20*np.pi/180), -20*np.pi/180)     # [-10,140]
            # q[23] = max(min(q[23], 20*np.pi/180), -20*np.pi/180)     # [-10,140]
            # # # Lumbar amd thorax twist
            # q[22] = max(min(q[22], 5*np.pi/180), -5*np.pi/180)     # [-10,140]
            # q[25] = max(min(q[25], 5*np.pi/180), -5*np.pi/180)     # [-10,140]
            # # Left Shoulder
            # q[42] = max(min(q[42], 150*np.pi/180), 0*np.pi/180)    # [0,150]
            # q[43] = max(min(q[43], 70*np.pi/180), -90*np.pi/180)    # [-70,90]
            # q[44] = max(min(q[44], 180*np.pi/180), -60*np.pi/180)    #  [-60,180]
            # # Right Shoulder
            # q[32] = max(min(q[32], 0*np.pi/180), -150*np.pi/180)    # [0,150]
            # q[33] = max(min(q[33], 90*np.pi/180), -70*np.pi/180)    # [-70,90]
            # q[34] = max(min(q[34], 180*np.pi/180), -60*np.pi/180)    #  [-60,180]
            # # elbow range of motion"
            # q[45] = max(min(q[45], 154*np.pi/180), -6*np.pi/180)    # [-6,154]
            # q[35] = max(min(q[35], 154*np.pi/180), -6*np.pi/180)    # [-6,154]
            
            self._nimble.setPositions(q)
            i+=1

    def qpIK(self,max_iterations=100,dt=None):
        i=0       
        if self.prob is None:         
            self.q = cp.Parameter(self.correct(np.array(self._nimble.getPositions())).shape)
            self.x = cp.Parameter(self.correct(np.array(self._nimble.getJointWorldPositions(self.joints))).shape)
            self.J = cp.Parameter(self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints).shape)
            self.x_target = cp.Parameter(self.x.shape)
            self.delta = cp.Variable(self.x.shape)
            self.dq = cp.Variable(self.q.shape)
            self.constraints = [self.x + self.J@self.dq == self.x_target + self.delta]  
            # self.constraints += [self.dq[21] == 0, self.dq[24] == 0, self.dq[29:32] == 0, self.dq[39:42] == 0 ]
            # self.constraints += [self.dq[21] == 0, self.dq[24] == 0]
            self.constraints += [-self.dq[6:] >= -1*(self.q_u[6:]-self.q[6:]), self.dq[6:] >= -1*(self.q[6:]-self.q_l[6:])]
            self.dq_prev = cp.Parameter(self.q.shape)

            # Velocity constraints
            self.dq_l = cp.Parameter(self.q.shape)
            self.dq_u = cp.Parameter(self.q.shape)
            self.constraints += [self.dq_prev[6:] + self.dq[6:] >= self.dq_l[6:], self.dq_prev[6:] + self.dq[6:] <= self.dq_u[6:]]
            
            self.obj = cp.Minimize( cp.quad_form(self.delta,np.eye(self.delta.shape[0])) + cp.quad_form(self.dq,np.eye(self.dq.shape[0])) )
            self.prob = cp.Problem(self.obj, self.constraints)
        
        self.dq_l.value = dt*self.qdot_l
        self.dq_u.value = dt*self.qdot_u
        self.dq_prev.value = np.zeros(self.q.shape)
        self.x_target.value = super().to_numpy(self.kps).reshape(1,-1).squeeze()
                
        older_loss = np.inf
        while i < max_iterations:
            self.q.value = self._nimble.getPositions()
            self.x.value = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
            self.J.value = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
            
            error = self.x.value - self.x_target.value
            loss = np.inner(error, error)
            # print(np.abs(older_loss - loss))
            if np.abs(older_loss - loss) < 0.0001:
                # print(i,self.prob.status,np.round(np.inner(self.x.value-self.x_target.value,self.x.value-self.x_target.value),2))
                break
            older_loss = loss
            
            self.prob.solve(solver=cp.ECOS)
            # print(self.dq.value)
            self.dq_prev.value += self.dq.value
            self._nimble.setPositions(self.q.value+self.dq.value) # *0.01
            
            i+=1
    
   
    
    def to_numpy(self):
        # return np.array(self._nimble.getJointWorldPositions(self.joints))
        return self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        