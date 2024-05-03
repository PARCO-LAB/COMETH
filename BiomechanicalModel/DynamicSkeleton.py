from .Skeleton import Skeleton,ConstrainedSkeleton
import nimblephysics as nimble
import torch
import numpy as np
import os
import cvxpy as cp

class Kalman():
    def __init__(self,dt,s,q=0.5):
        self.X = np.array([[s],[0.1],[0.01]])
        self.P = np.diag((1, 1, 1))
        self.F = np.array([[1, dt, dt*dt/2], [0, 1, dt], [0, 0, 1]])
        self.Q = np.eye(self.X.shape[0])*q
        self.Y = np.array([s])
        self.H = np.array([1, 0, 0]).reshape(1,3)
        self.R = 1 # np.eye(self.Y.shape[0])*
        
    def predict(self,dt = None):
        if dt is not None:
            self.F = np.array([[1, dt, dt*dt/2], [0, 1, dt], [0, 0, 1]])
        self.X = np.dot(self.F,self.X) #+ np.dot(self.B,self.U)
        self.P = np.dot(self.F, np.dot(self.P,self.F.T)) + self.Q

    def update(self,Y,R=None,minval=-np.inf,maxval=np.inf):
        self.Y = Y
        if R is not None:
            self.R = R
        self.K = np.dot(self.P,self.H.T) / ( self.R + np.dot(self.H,np.dot(self.P,self.H.T)) ) 
        self.X = self.X + self.K * ( Y - np.dot(self.H,self.X))
        self.P = np.dot((np.eye(self.X.shape[0])- np.dot(self.K,self.H)),self.P)
        self.X[0] = max(min(self.X[0],maxval),minval)
        self.Y = float(np.dot(self.H,self.X))
        return self.get_output()

    def get_output(self):
        return float(np.dot(self.H,self.X))

class DynamicSkeleton(ConstrainedSkeleton):
    def __init__(self, config, name=None, osim_file=None, geometry_dir='', max_velocity=5):
        
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

        
        self.neutral_position = self._nimble.getPositions()
        s_avg = (self.q_l + self.q_u) / 2
        self.neutral_position[6:] = s_avg[6:]
        
        # self.qdot_l = np.array([-0.55,-0.43,-1.04,-0.74,-0.20,-0.30,-1.58,-0.57,-0.61,-1.97,0,0,0,-1.61,-0.56,-0.55,-1.97,0,0,0,-0.49,0,-0.31,-0.37,0,-0.29,0,0,0,0,0,0,-0.80,-0.84,-1.37,-1.34,-0.080,0,0,0,0,0,-0.80,-0.84,-1.42,-1.16,-0.070,0,0])
        # self.qdot_u = np.array([0.57,0.43,0.95,0.84,0.20,0.29,1.93,0.54,0.53,2.14,0,0,0,1.95,0.54,0.51,2.23,0,0,0,0.49,0,0.32,0.38,0,0.29,0,0,0,0,0,0,0.78,0.88,1.4,1.47,0.090,0,0,0,0,0,0.84,0.72,1.37,1.27,0.080,0,0])
        
        self.qdot_l = np.zeros(self.q_u.shape)-max_velocity
        self.qdot_u = np.zeros(self.q_u.shape)+max_velocity
        
        self.prob = None
        self.prev_mask = None
        self.kf = None
        self.joints = [self._nimble.getJoint(l) for l in nimble_joint_names]
        pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        self.s12_base.load_from_numpy(pos.reshape(-1,3),self.kps)
        self.skeleton_from_nimble.load_from_BODY12(self.s12_base)
        
    def reset_history(self):
        for b in self.bones_list:
            b.history = []
        for kp in self.keypoints_list:
            kp._history = []
        self.height_history = []
    
    def reset(self):
        self._nimble.setPositions(self.neutral_position)
        self.reset_history()
        self.kf = None
    
    def estimate_confidence(self):
        # Update each keypoint.confidence value
        h = np.nanmean(self.height_history)
        for b in self.bones_list:
            min_l = (self.proportions[b.name][0]-2*self.proportions[b.name][1])*self.height_history[-1]
            max_l = (self.proportions[b.name][0]+2*self.proportions[b.name][1])*self.height_history[-1]
            # If they are in range, increase confidence
            if b.length > min_l and b.length < max_l:
                    b.src.confidence  = min( b.src.confidence + 0.1, 1) 
                    b.dest.confidence  = min( b.dest.confidence + 0.1, 1) 
            else:
                    b.src.confidence  = max( b.src.confidence - 0.1, 0) 
                    b.dest.confidence  = max( b.dest.confidence - 0.1, 0)
        
    
    # Remove the joint position to place the corrected hip from marker-based
    def correct(self,pos):
        # Correct the pelvis joint
        transform = self._nimble.getBodyNode('pelvis').getWorldTransform()
        scale = self._nimble.getBodyNode('pelvis').getScale()
        pos[3*self.kps.index("RHip"):3*self.kps.index("RHip")+3] = transform.multiply(np.multiply(scale,self.RHip))
        pos[3*self.kps.index("LHip"):3*self.kps.index("LHip")+3] = transform.multiply(np.multiply(scale,self.LHip))
        
        # # Correct the scapula joints
        transform = self._nimble.getBodyNode('scapula_l').getWorldTransform()
        pos[3*self.kps.index("LShoulder"):3*self.kps.index("LShoulder")+3] = transform.multiply(self.LShoulder)
        transform = self._nimble.getBodyNode('scapula_r').getWorldTransform()
        pos[3*self.kps.index("RShoulder"):3*self.kps.index("RShoulder")+3] = transform.multiply(self.RShoulder)
        return pos
    
    # Scaling better suited for noisy input (e.g., marker-less data)
    def estimate_scale(self):
        scale =  self._nimble.getBodyScales().reshape(-1,3)
        # If there may be error is the height and bones estimation, return the mean of the previous
        if np.all(np.isnan(self.height_history)):
            return
        # print("here")
        h = np.nanmean(self.height_history)
        for i,b in enumerate(self.body_dict.keys()):
            if self.body_dict[b] == 'Core' or self.body_dict[b] == '':
                scale[i,:] = h / self.skeleton_from_nimble.estimate_height()
            else:
                sc = np.nanmean(self.bones_dict[self.body_dict[b]].history) / self.skeleton_from_nimble.bones_dict[self.body_dict[b]].length
                # If there is a symmetrical one
                if self.body_dict[b] in self.symmetry:
                    if np.isnan(sc):
                        sc = np.nanmean(self.bones_dict[self.symmetry[self.body_dict[b]]].history) / self.skeleton_from_nimble.bones_dict[self.symmetry[self.body_dict[b]]].length
                    else:
                        sc_sym = np.nanmean(self.bones_dict[self.symmetry[self.body_dict[b]]].history) / self.skeleton_from_nimble.bones_dict[self.symmetry[self.body_dict[b]]].length
                        if np.abs(1-sc) > np.abs(1-sc_sym): 
                            sc = sc_sym
                            # print("symmetric law for",b)
                if not np.isnan(sc):
                    scale[i,:] = sc
        
        # Clip the scaling between fixed bounds
        # scale = np.clip(scale,0.85,1.15)
        avg_scale = np.mean(scale)
        scale = np.clip(scale,avg_scale-0.05,avg_scale+0.05)
        
        self._nimble.setBodyScales(scale.reshape(-1,1))


    # Old scaling version, only for precise input (e.g., marker-based)
    def scale(self):
        scale =  self._nimble.getBodyScales().reshape(-1,3)
        # If there may be error is the height and bones estimation, return the mean of the previous
        if np.all(np.isnan(self.height_history)):
            return
        h = np.nanmean(self.height_history)
        for i,b in enumerate(self.body_dict.keys()):
            if self.body_dict[b] == 'Core' or self.body_dict[b] == '':
                scale[i,:] = h / self.skeleton_from_nimble.estimate_height()
            else:
                sc = np.nanmean(self.bones_dict[self.body_dict[b]].history) / self.skeleton_from_nimble.bones_dict[self.body_dict[b]].length
                if np.isnan(sc) and self.body_dict[b] in self.symmetry:
                    sc = np.nanmean(self.bones_dict[self.symmetry[self.body_dict[b]]].history) / self.skeleton_from_nimble.bones_dict[self.symmetry[self.body_dict[b]]].length
                if not np.isnan(sc):
                    scale[i,:] = sc
        # print(np.round(scale[:,0].transpose(),2))
        self._nimble.setBodyScales(scale.reshape(-1,1))
            
    
    # Inverse kinematics through gradient descend
    def exact_scale(self,max_iterations=1000,precision=0.001):
        older_loss = np.inf
        mask = ~np.isnan(super().to_numpy(self.kps)) 
        target = super().to_numpy(self.kps)[mask].reshape(1,-1).squeeze()
        for _ in range(max_iterations):
            # Angular position placement
            q = self._nimble.getPositions()
            i=0
            while i < max_iterations:
                pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
                # pos = np.array(self._nimble.getJointWorldPositions(self.joints))
                pos = pos[mask.reshape(1,-1).squeeze()]
                d_loss_d__pos = 2 * (pos - target)
                d_pos_d_joint_angles = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
                d_pos_d_joint_angles = d_pos_d_joint_angles[mask.reshape(1,-1).squeeze(),:]
                d_loss_d_joint_angles = d_pos_d_joint_angles.T @ d_loss_d__pos
                q -= 0.05 * d_loss_d_joint_angles            
                q = np.clip(q,self.q_l,self.q_u)
                self._nimble.setPositions(q)
                i+=1
            
            # Scaling setting
            scale =  self._nimble.getBodyScales()
            j = 0
            while j < max_iterations:
                pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
                # pos = np.array(self._nimble.getJointWorldPositions(self.joints))
                pos = pos[mask.reshape(1,-1).squeeze()]
                d_loss_d__pos = 2 * (pos - target)
                d_pos_d_scales = self._nimble.getJointWorldPositionsJacobianWrtBodyScales(self.joints)
                d_pos_d_scales = d_pos_d_scales[mask.reshape(1,-1).squeeze(),:]
                # d_pos_d_scales = d_pos_d_scales[mask.reshape(1,-1).squeeze(),0:72:3]
                d_loss_d_scales = d_pos_d_scales.T @ d_loss_d__pos
                # d_loss_d_scales = d_loss_d_scales.reshape((1,-1))
                # d_loss_d_scales = np.array([d_loss_d_scales,d_loss_d_scales,d_loss_d_scales]).transpose()
                # d_loss_d_scales = d_loss_d_scales.squeeze().reshape((-1,))
                scale -= 0.001 * d_loss_d_scales
                self._nimble.setBodyScales(scale)
                j+=1

            error = np.array(self._nimble.getJointWorldPositions(self.joints))[mask.reshape(1,-1).squeeze()] - target
            loss = np.inner(error, error)
            if np.abs(older_loss - loss) < precision:
                print(loss)
                break
            older_loss = loss
            





    def qpIK(self,max_iterations=100,dt=0.02,precision=0.00001):
        mask = ~np.isnan(super().to_numpy(self.kps))
        subset_joints = [self.joints[i] for i in range(len(self.joints)) if mask[i,0]]
        
        x_target = super().to_numpy(self.kps)[mask].reshape(1,-1).squeeze()
        
        if np.any(mask != self.prev_mask):
            self.prob = None        
        i=0
        
        self.prev_mask = mask
        
        if self.prob is None:        
            # self.reset() 
            dt = 100
            self.q = cp.Parameter(np.array(self._nimble.getPositions()).shape)
            self.x = cp.Parameter(np.array(self._nimble.getJointWorldPositions(subset_joints)).shape)
            self.J = cp.Parameter(self._nimble.getJointWorldPositionsJacobianWrtJointPositions(subset_joints).shape)
            self.x_target = cp.Parameter(self.x.shape)
            self.delta = cp.Variable(self.x.shape)
            # self.Rdiag = cp.Parameter(self.x.shape, nonneg=True)
            # self.Rdiag = cp.Parameter((self.x.shape[0],self.x.shape[0]), PSD = True)
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
            # # self.obj = cp.Minimize( cp.quad_form(self.delta,cp.diag(self.Rdiag)) + cp.quad_form(self.dq,np.eye(self.dq.shape[0])) )
            # self.obj = cp.Minimize( cp.quad_form(self.delta,self.Rdiag) + cp.quad_form(self.dq,np.eye(self.dq.shape[0])) )
            self.prob = cp.Problem(self.obj, self.constraints)
            
        
        # self.Rdiag.value = np.diag([self.keypoints_dict[kp].confidence for kp in self.kps for _ in range(3)])
                
        self.dq_l.value = dt*self.qdot_l
        self.dq_u.value = dt*self.qdot_u
        self.dq_prev.value = np.zeros(self.q.shape)
        self.x_target.value = x_target
        
                
        older_loss = np.inf
        while i < max_iterations:
            self.q.value = self._nimble.getPositions()
            x = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
            J = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
            self.J.value = J[mask.reshape(1,-1).squeeze(),:]
            self.x.value = x[mask.reshape(1,-1).squeeze()]
            
            error = self.x.value - self.x_target.value
            loss = np.inner(error, error)
            if np.abs(older_loss - loss) < precision:
                break
            older_loss = loss
            
            self.prob.solve(solver=cp.ECOS)
            # print(i,self.prob.status,type(self.dq.value))
            self.dq_prev.value += np.array(self.dq.value)
            self._nimble.setPositions(self.q.value+self.dq.value) # *0.01
            
            i+=1
        
    def filter(self,dt,Q=0.5):
        if self.kf is None:
            self.qpIK(100,0.02,precision=0.0001)
            pos = self._nimble.getPositions()
            self.kf = [Kalman(dt,pos[i],Q) for i in range(pos.shape[0])]
        else:
            [kf.predict() for kf in self.kf]
            self.qpIK(100,0.02,precision=0.0001)
            pos = self._nimble.getPositions()
            for i in range(len(self.kf)):
                pos[i] = self.kf[i].update(pos[i],minval=self.q_l[i],maxval=self.q_u[i])
            # if not (np.all(pos>=self.q_l) and np.all(pos<=self.q_u)):
            #     print(pos>=self.q_l)
            #     print(pos<=self.q_u)
            self._nimble.setPositions(pos)
            
            
    def to_numpy(self):
        # return np.array(self._nimble.getJointWorldPositions(self.joints))
        return self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        