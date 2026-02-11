from typing import Dict, Tuple
from .Skeleton import Skeleton,ConstrainedSkeleton
# import COMETH.parameters as COMETH_parameters
from . import parameters as COMETH_parameters
from .parameters import *
import nimblephysics as nimble
import torch
import numpy as np
import os, sys
import cvxpy as cp
current_path = os.path.dirname(os.path.abspath(__file__)) + "/"

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

# template to use from file reading for insane speedup
template_skeleton: Dict[str, Tuple[nimble.dynamics.Skeleton, str]] = {}

class DynamicSkeleton(ConstrainedSkeleton):
    def __init__(self, config=current_path+'BODY15_constrained_3D.xml', name=None, osim_file=None, geometry_dir='', max_velocity=None):
        
        super().__init__(config, name)
        self.timestamp = 0
        self.last_timestamp = 0
        self.keypoints_dict = {obj.name: obj for obj in self.keypoints_list}

        if osim_file not in template_skeleton:
            if osim_file is not None:
                rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(osim_file,geometry_dir)
                self.type = 'BSM'
            else:
                rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(current_path+"bsm.osim")
                self.type = 'BSM'
            self._nimble: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton
            # cache it
            template_skeleton[osim_file] = (rajagopal_opensim.skeleton.clone(), self.type)
        else:
            cache = template_skeleton[osim_file]
            self._nimble: nimble.dynamics.Skeleton = cache[0].clone()
            self.type = cache[1]
            
        self.measurements = []
            
        # Offsets of hip and shoulder, usefull for the correction process
        self.RShoulder = (COMETH_parameters.RCAJ_OFFSET+COMETH_parameters.RHGT_OFFSET)/2
        self.LShoulder = (COMETH_parameters.LCAJ_OFFSET+COMETH_parameters.LHGT_OFFSET)/2
        self.RHip = (COMETH_parameters.RASI_OFFSET+COMETH_parameters.RPSI_OFFSET)/2
        self.LHip = (COMETH_parameters.LASI_OFFSET+COMETH_parameters.LPSI_OFFSET)/2        
        
        self.s12_base = Skeleton(current_path+'BODY12.xml')
        self.s15_base = ConstrainedSkeleton(current_path+'BODY15_constrained_3D.xml')
        self.skeleton_from_nimble = ConstrainedSkeleton(current_path+'BODY15_constrained_3D.xml')
        
        if  self.type == 'rajagopal':
            self.kps =  COMETH_parameters.RAJAGOPAL_KPS
            nimble_joint_names = COMETH_parameters.RAJAGOPAL_JOINT_NAMES
            self.bodies_name_translated = COMETH_parameters.RAJAGOPAL_BODIES_NAME_TRANSLATED
        elif self.type == "BSM":
            self.kps =  COMETH_parameters.BSM_KPS
            nimble_joint_names = COMETH_parameters.BSM_JOINT_NAMES
            self.bodies_name_translated = COMETH_parameters.BSM_BODIES_NAME_TRANSLATED

        self.hip_correction = True  # manually adjust the hip offset (disable if ur using marker-based mocap)
        # ----------------------------------------------------------------------
        # For IMUs
        # self.bodies_dict = {obj.getName(): obj for obj in self._nimble.getBodyNodes()}
        self.IMUs = []
        
        #-----------------------------------------------------------------------
        
        # Turn the limits in radians
        self.q_l = COMETH_parameters.Q_LOWER_BOUND*np.pi/180
        self.q_u = COMETH_parameters.Q_UPPER_BOUND*np.pi/180

        # Set the initial position in between the limits 
        # Root initial position [xrot?,yrot?,zrot?,xpos,ypos,zpos]
        self.neutral_position = self._nimble.getPositions()
        self.neutral_position[:6] = np.zeros(6)
        self.neutral_position[0] = np.pi
        self.neutral_position[2] = -np.pi/2
        # Joints initial position
        s_avg = (self.q_l[6:] + self.q_u[6:]) / 2
        self.neutral_position[6:] = s_avg
        
        # Set the velocity limits
        if max_velocity is None:
            # Acquired from BSM dataset
            self.qdot_l = COMETH_parameters.QDOT_LOWER_BOUND
            self.qdot_u = COMETH_parameters.QDOT_UPPER_BOUND
        else:
            self.qdot_l = np.zeros(self.q_u.shape)-max_velocity
            self.qdot_u = np.zeros(self.q_u.shape)+max_velocity
        
        
        self.prob = None
        self.prev_mask = None
        self.kf = None
        self.joints = [self._nimble.getJoint(l) for l in nimble_joint_names]
        if self.hip_correction:
            pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        else:
            pos = np.array(self._nimble.getJointWorldPositions(self.joints))
        self.s12_base.load_from_numpy(pos.reshape(-1,3),self.kps)
        self.skeleton_from_nimble.load_from_BODY12(self.s12_base)
    
        # Save for faster qpIK
        self.qpIK_problems = {}
        self.reset()
        
        
    def reset_history(self):
        for b in self.bones_list:
            b.history = []
        for kp in self.keypoints_list:
            kp._history = []
        self.height_history = []
        self.measurements = []
    
    def reset(self):
        self._nimble.setPositions(self.neutral_position)
        self.reset_history()
        self.kf = None
        # self._nimble.setBodyScales()
    
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
        
    
    # Remove the joint position to place the corrected hip (closer to ASI and PSI)
    def correct(self,pos):
        
        # # Old version (some of the nimble's features are not working with torch >2 -> segfault)
        # transform = self._nimble.getBodyNode('pelvis').getWorldTransform()
        # pos[3*self.kps.index("RHip"):3*self.kps.index("RHip")+3] = transform.multiply(np.multiply(scale,self.RHip))
        # pos[3*self.kps.index("LHip"):3*self.kps.index("LHip")+3] = transform.multiply(np.multiply(scale,self.LHip))
        # transform = self._nimble.getBodyNode('scapula_l').getWorldTransform()
        # pos[3*self.kps.index("LShoulder"):3*self.kps.index("LShoulder")+3] = transform.multiply(self.LShoulder)
        # transform = self._nimble.getBodyNode('scapula_r').getWorldTransform()
        # pos[3*self.kps.index("RShoulder"):3*self.kps.index("RShoulder")+3] = transform.multiply(self.RShoulder)
        
        # Correct the pelvis joint
        scale = self._nimble.getBodyNode('pelvis').getScale()
        transform = self._nimble.getBodyNode('pelvis').getWorldTransform().matrix()
        point = np.hstack((np.multiply(scale,self.RHip),1))
        pos[3*self.kps.index("RHip"):3*self.kps.index("RHip")+3] = transform.dot(point)[:-1]
        
        point = np.hstack((np.multiply(scale,self.LHip),1))
        pos[3*self.kps.index("LHip"):3*self.kps.index("LHip")+3] = transform.dot(point)[:-1]
        
        # # Correct the scapula joints
        transform = self._nimble.getBodyNode('scapula_l').getWorldTransform().matrix()
        point = np.hstack((self.LShoulder,1))
        pos[3*self.kps.index("LShoulder"):3*self.kps.index("LShoulder")+3] = transform.dot(point)[:-1]
        
        transform = self._nimble.getBodyNode('scapula_r').getWorldTransform().matrix()
        point = np.hstack((self.RShoulder,1))
        pos[3*self.kps.index("RShoulder"):3*self.kps.index("RShoulder")+3] = transform.dot(point)[:-1]
        
        return pos
    
    # After the scaling process, if there are measurements too far from the 
    # skeleton, remove those keypoints
    def remove_outlier_measurements(self,mapping):
        
        # Set the current 3D joint position of the skeleton if 
        # never gone trhough qpIK
        if np.isnan(self.keypoints_dict["Root"].pos[0]):
            if self.hip_correction:
                pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints))).reshape(-1,3)
            else:
                pos = np.array(self._nimble.getJointWorldPositions(self.joints)).reshape(-1,3)
            self.s12_base.load_from_numpy(pos.reshape(-1,3),self.kps)
            self.load_from_BODY12(self.s12_base)
        
        # For each measurement, remove the keypoints whose bone is too long for
        # the height of the skeleton
        for i in range(len(self.measurements)-1,-1,-1):
            m = self.measurements[i][mapping,:]
            self.s12_base.load_from_numpy(m,self.kps)
            self.s15_base.load_from_BODY12(self.s12_base)
            for j,b in enumerate(self.s15_base.bones_list):               
                if  b.length > COMETH_parameters.MAX_BONE_LENGTH or \
                    abs(self.bones_dict[b.name].length-b.length) > COMETH_parameters.MAX_BONE_LENGTH_DIFFERENCE:
                    print("Removed", b.dest.name)
                    # self.s15_base.bones_list[j].src.pos = np.array([np.nan,np.nan,np.nan])
                    self.s15_base.bones_list[j].dest.pos = np.array([np.nan,np.nan,np.nan])
            # Dump back the keypoints into the measurements format
            m = self.s15_base.to_numpy(self.kps)
            
            # print(m)
            # Remove out of cluster measurements
            distances = np.linalg.norm(m - np.nanmean(m, axis=0), axis=1)
            threshold = max(np.nanmean(distances) + 2 * np.nanstd(distances), COMETH_parameters.MAX_DISTANCE_FROM_CLUSTER)
            m[distances > threshold, :] = np.nan
            # print(threshold)
            # print(m)
            # exit()
            
            if np.all(np.isnan(m)):
                self.measurements.pop(i)
            else:
                self.measurements[i][mapping,:] = m
    
    
    
    # Scaling better suited for noisy input (e.g., marker-less data)
    def estimate_scale(self):
        scale =  self._nimble.getBodyScales().reshape(-1,3)
        # If there may be error is the height and bones estimation, return the mean of the previous
        if np.all(np.isnan(self.height_history)):
            return

        h = np.nanmean(self.height_history) # Old height from previous frames

        for i,b in enumerate(self.bodies_name_translated.keys()):
            if self.bodies_name_translated[b] == 'Core' or self.bodies_name_translated[b] == '':
                scale[i,:] = h / self.skeleton_from_nimble.estimate_height()
            else:
                sc = np.nanmean(self.bones_dict[self.bodies_name_translated[b]].history) / self.skeleton_from_nimble.bones_dict[self.bodies_name_translated[b]].length
                # If there is a symmetrical one
                if self.bodies_name_translated[b] in self.symmetry:
                    if np.isnan(sc):
                        sc = np.nanmean(self.bones_dict[self.symmetry[self.bodies_name_translated[b]]].history) / self.skeleton_from_nimble.bones_dict[self.symmetry[self.bodies_name_translated[b]]].length
                    else:
                        sc_sym = np.nanmean(self.bones_dict[self.symmetry[self.bodies_name_translated[b]]].history) / self.skeleton_from_nimble.bones_dict[self.symmetry[self.bodies_name_translated[b]]].length
                        if np.abs(1-sc) > np.abs(1-sc_sym): 
                            sc = sc_sym
                if not np.isnan(sc):
                    scale[i,:] = sc
        
        # V0: Clip the scaling between fixed bounds ----------------------------
        # avg_scale = np.mean(scale)
        # scale = np.clip(scale,avg_scale-0.05,avg_scale+0.05)
        # self._nimble.setBodyScales(scale.reshape(-1,1))
        
        # V1: Clip the scaling between fixed bounds ----------------------------
        avg_scale = np.mean(scale)
        if avg_scale > 0.7 and avg_scale < 1.3:
            avg_scale = np.mean(scale)
        else:
            avg_scale = h / self.skeleton_from_nimble.estimate_height()
        # ----------------------------------------------------------------------            
        scale = np.clip(scale,avg_scale-0.05,avg_scale+0.05) # A skeleton may not have the same proportions as the BSM (5%)
        self._nimble.setBodyScales(scale.reshape(-1,1))
            
    # Inverse kinematics through gradient descend
    def exact_scale(self,max_iterations=1000,precision=0.001,to_scale=True):
        older_loss = np.inf
        mask = ~np.isnan(super().to_numpy(self.kps)) 
        target = super().to_numpy(self.kps)[mask].reshape(1,-1).squeeze()
        for _ in range(max_iterations):
            # Angular position placement
            q = self._nimble.getPositions()
            i=0
            while i < max_iterations:
                if self.hip_correction:
                    pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
                else:
                    pos = np.array(self._nimble.getJointWorldPositions(self.joints))
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
            if to_scale:
                scale =  self._nimble.getBodyScales()
                j = 0
                while j < max_iterations:
                    if self.hip_correction:
                        pos = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
                    else:
                        pos = np.array(self._nimble.getJointWorldPositions(self.joints))
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
                # print(loss)
                break
            older_loss = loss
    
    # multisource_qpIK from 3D keypoints targets
    def qpIK(self,targets,max_iterations=100,dt=0.02,precision=0.00001):
        
        # Get the number of keypoints seen from each source and sort them from
        # the lowest to the highest
        masks = [~np.isnan(t) for t in targets]
        nkey = np.sort([int(np.sum(m[:,0])) for m in masks])
        permutation = np.argsort([int(np.sum(m[:,0])) for m in masks])
        targets = [targets[i] for i in permutation]
        masks = [masks[i] for i in permutation]
        
        # Generate the key to access the dictionary (for performance)
        key = ""
        for k in nkey.tolist():
            key+=str(k)+"."
        
        # TODO: remove?
        # print(key,key in self.qpIK_problems.keys())
        
        problem_to_build = False if key in self.qpIK_problems.keys() else True
            
        subsets_joints = []
        for mask in masks:
            subsets_joints.append([self.joints[i] for i in range(len(self.joints)) if mask[i,0]])
                
        if np.all(self._nimble.getPositions() == self.neutral_position):
            dt = 100
        
        # Every time set a new problem. It's slower but can be improved
        if problem_to_build:
            self.q = cp.Parameter((49,))
            self.xs = [cp.Parameter((nk*3,)) for nk in nkey]
            self.Js = [cp.Parameter((nk*3,49)) for nk in nkey]
            self.x_targets = [cp.Parameter((nk*3,)) for nk in nkey]
            self.deltas =  [cp.Variable((nk*3,)) for nk in nkey]
            self.dq = cp.Variable((49,))
            self.constraints = []
            self.constraints += [self.xs[i] + self.Js[i]@self.dq == self.x_targets[i] + self.deltas[i] for i in range(len(masks))]
            # Joint limits
            # self.constraints += [-self.dq[6:] >= -1*(self.q_u[6:]-self.q[6:]), self.dq[6:] >= -1*(self.q[6:]-self.q_l[6:])]
            self.constraints += [-self.dq >= -1*(self.q_u-self.q), self.dq >= -1*(self.q-self.q_l)]
            self.dq_prev = cp.Parameter((49,))
            # Velocity constraints
            self.dq_l = cp.Parameter((49,))
            self.dq_u = cp.Parameter((49,))
            # self.constraints += [self.dq_prev[6:] + self.dq[6:] >= self.dq_l[6:], self.dq_prev[6:] + self.dq[6:] <= self.dq_u[6:]]
            self.constraints += [self.dq_prev + self.dq >= self.dq_l, self.dq_prev + self.dq <= self.dq_u]
            to_minimize = cp.quad_form(self.dq,np.eye(self.dq.shape[0]))
            for delta in self.deltas:
                to_minimize += cp.quad_form(delta,np.eye(delta.shape[0]))
            self.obj = cp.Minimize(to_minimize)
            self.prob = cp.Problem(self.obj, self.constraints)
            self.qpIK_problems[key] = {"problem": self.prob, 
                                                   "x_targets":self.x_targets,
                                                   "xs" : self.xs,
                                                   "Js" : self.Js,
                                                   "deltas" : self.deltas,
                                                   "dq_l" : self.dq_l,
                                                   "dq_u" : self.dq_u,
                                                   "dq_prev" : self.dq_prev,
                                                   "dq" : self.dq,
                                                   "q" : self.q
                                                   }
        else:
            self.prob = self.qpIK_problems[key]["problem"]
            self.x_targets = self.qpIK_problems[key]["x_targets"]
            self.xs = self.qpIK_problems[key]["xs"]
            self.Js = self.qpIK_problems[key]["Js"]
            self.deltas = self.qpIK_problems[key]["deltas"]
            self.dq_l = self.qpIK_problems[key]["dq_l"]
            self.dq_u = self.qpIK_problems[key]["dq_u"]
            self.dq_prev = self.qpIK_problems[key]["dq_prev"]
            self.dq = self.qpIK_problems[key]["dq"]
            self.q = self.qpIK_problems[key]["q"]
        
        # self.Rdiag.value = np.diag([self.keypoints_dict[kp].confidence for kp in self.kps for _ in range(3)])
                
        self.dq_l.value = dt*self.qdot_l
        self.dq_u.value = dt*self.qdot_u
        self.dq_prev.value = np.zeros(self.q.shape)
        for i,x_target in enumerate(self.x_targets):
            x_target.value = targets[i][masks[i]]
        
        i = 0
        older_loss = np.inf
        while i < max_iterations:
            self.q.value = self._nimble.getPositions()
            # x is the position of the skeleton joints in the 3D, J the jacobian
            if self.hip_correction:
                x = self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
            else:
                x = np.array(self._nimble.getJointWorldPositions(self.joints))
            J = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)
            # For each target, update the values of the jacobians and the starting position
            for j in range(len(self.x_targets)):
                self.Js[j].value = J[masks[j].reshape(1,-1).squeeze(),:]
                self.xs[j].value = x[masks[j].reshape(1,-1).squeeze()]
            
            # V0-1-2 Loss
            # error = np.nanmean([self.xs[j].value - self.x_targets[j].value])
            
            # V3 Loss
            error = np.nanmean([np.nanmean(self.xs[j].value - self.x_targets[j].value) for j in range(len(self.x_targets))])
            
            # If the error loss is stationary, exit
            loss = np.inner(error, error)
            if np.abs(older_loss - loss) < precision:
                break
            older_loss = loss
            
            # self.prob.solve(solver=cp.ECOS)
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            # print(i,self.prob.status,type(self.dq.value))
            self.dq_prev.value += np.array(self.dq.value)
            self._nimble.setPositions(self.q.value+self.dq.value) # *0.01
            i+=1
    
    # Use a Kalman filter and the qpIK to smooth the data and move the skeleton towards measurements
    def filter(self,data_list=None,iterations=COMETH_parameters.QPIK_ITERATIONS,dt=100,Q=0.001,to_predict=True, precision=COMETH_parameters.QPIK_PRECISION):
        self.qpIK(data_list,iterations,dt,precision=precision)
        pos = self._nimble.getPositions()
        
        # Check if the Kalman filter has been initialized
        if self.kf is None:
            self.kf = [Kalman(dt,pos[i],Q) for i in range(pos.shape[0])]
        else:
            if to_predict:
                [kf.predict() for kf in self.kf]
            for i in range(len(self.kf)):
                pos[i] = self.kf[i].update(pos[i],minval=self.q_l[i],maxval=self.q_u[i])
            self._nimble.setPositions(pos)
            
    def to_numpy(self):
        if self.hip_correction:
            return self.correct(np.array(self._nimble.getJointWorldPositions(self.joints)))
        else:
            return np.array(self._nimble.getJointWorldPositions(self.joints))

    def getImuWorldOrient(self):
        """ 
        Returns the IMUs transform matrix in world frame as a numpy array of size (Nx4x4) with N = number of IMUs.
        """
        rot = []
        for imu in self.IMUs:
            T_b_w = imu[0].getWorldTransform().matrix()
            T_i_b = imu[1].matrix()

            T_i_w = T_b_w @ T_i_b

            rot.append(T_i_w)
        
        return np.array(rot)

    def getImuWorldPosition(self):
        """ 
        Returns the IMUs positions in world frame as a numpy array of size (Nx3) with N = number of IMUs.
        """
        pos = []
        for imu in self.IMUs:
            T_b_w = imu[0].getWorldTransform().matrix()
            T_i_b = imu[1].matrix()

            T_i_w = T_b_w @ T_i_b
            pos.append(T_i_w[:3,3].reshape(3))
            
        return np.array(pos)

    def correctImuOrient(self, R_corr):
        """ Apply to previous IMUs rotation a new rotation in nimble model. 
        R_corr is (Nx3x3) numpy array with N = number of IMUs
        """
        assert len(R_corr)==len(self.IMUs)

        for i, imu in enumerate(self.IMUs):
            T_i_b = imu[1].matrix()
            T = np.eye(4)
            T[:3,:3]= R_corr[i]

            T_corr = T@T_i_b

            self.IMUs[i] = (imu[0],nimble.math.Isometry3(T_corr[:3,:3], T_corr[:3,3]))

    def setImuWorldTransform(self, T_i_w):
        """ Set new rotation for each IMU in the nimble model.
        T_i_w is the transformation matrix in world frame with shape (N,4,4) with N = number of IMUs.
        Only the rotation of transform matrix is applied.
        """
        
        for i, imu in enumerate(self.IMUs):
            T_b_w = imu[0].getWorldTransform().matrix()
            T_old = imu[1].matrix()
            T_i_b = np.linalg.inv(T_b_w)@T_i_w[i]
            self.IMUs[i] = (imu[0],nimble.math.Isometry3(T_i_b[:3,:3], T_old[:3,3]))

    def qpik_IMU(self, acc_target=None, gyro_target=None, position_target=None, max_iterations=100, precision=0.0001, dt=1/60):
        # Variable declaration
        dq = cp.Variable((49,))
        delta_a =  cp.Variable((9,))
        delta_w =  cp.Variable((9,))
        dq_cumulative = cp.Variable((49,))
        
        # Acc
        if acc_target is not None:
            a = cp.Parameter((9,))
            aT = cp.Parameter((9,)) 
            Ja = cp.Parameter((9,49))
        
        # Gyro
        if gyro_target is not None:
            w = cp.Parameter((9,))
            wT = cp.Parameter((9,)) 
            Jw = cp.Parameter((9,49))

        # Pos from HPE
        if position_target is not None:
            mask = np.array([~np.isnan(t) for t in position_target])
            nkey = int(np.sum(mask[:,0]))
            subsets_joints = [([position_target[i] for i in range(len(position_target)) if mask[i,0]])]
            x = cp.Parameter((nkey*3))
            xT = cp.Parameter((nkey*3))
            Jp = cp.Parameter((nkey*3,49))
            delta_p = cp.Variable((nkey*3,))
        else:
            delta_p = None
        
        # Position
        q = cp.Parameter((49,))
        q_l = cp.Parameter((49,))
        q_u = cp.Parameter((49,))
        dq_l = cp.Parameter((49,))
        dq_u = cp.Parameter((49,))

        
        # Problem building
        constraints = []
        if acc_target is not None:
            constraints += [a + Ja@dq + delta_a == aT]
        
        if gyro_target is not None:
            constraints += [w + 1*Jw@dq + delta_w == wT]

        if position_target is not None:
            constraints += [x + Jp@dq + delta_p== xT]
        
        dq_cumulative.value = np.zeros(q.shape)
        
        constraints += [dq <= q_u - q] # upper position constraint
        constraints += [dq >= q_l - q] # lower position constraint
        constraints += [dq_cumulative + dq >= dq_l, dq_cumulative + dq <= dq_u]  # velocity constraint
    
                            
        lambda_reg  = 10
        weight_acc  = 100 if acc_target is not None else 0.0
        weight_gyro = .01 if gyro_target is not None else 0.0
        weight_hpe = 10000 if position_target is not None else 0.0

        # Reducing accelerometer weight if target position is given
        if acc_target is not None and position_target is not None:
            weight_acc = 1

        to_minimize = (
            lambda_reg * cp.sum_squares(dq) +
            weight_acc * cp.sum_squares(delta_a) +
            weight_gyro * cp.sum_squares(delta_w) +
            weight_hpe * cp.sum_squares(delta_p)
        )

        obj = cp.Minimize(to_minimize)
        prob = cp.Problem(obj, constraints)
        
        q_l.value = np.clip(self.q_l,-1000,1000)
        q_u.value = np.clip(self.q_u,-1000,1000)
        dq_l.value = dt*self.qdot_l
        dq_u.value = dt*self.qdot_u

        # No speed limit in case of positional target
        if position_target is not None:
            dq_l.value = dt*1000*self.qdot_l
            dq_u.value = dt*1000*self.qdot_u
        i=0
        
        # Solving the QP iterativly
        while i < max_iterations:
            i+=1
            # Parameters initialization
            q.value = self._nimble.getPositions()

            if acc_target is not None:
                aT.value = acc_target
                a.value = self._nimble.getAccelerometerReadings(self.IMUs)
                Ja.value = self._nimble.getAccelerometerReadingsJacobianWrt(accs=self.IMUs, wrt=nimble.neural.WRT_POSITION)
            
            if gyro_target is not None:
                wT.value = gyro_target
                w.value = self._nimble.getGyroReadings(self.IMUs)
                Jw.value = self._nimble.getGyroReadingsJacobianWrt(gyros=self.IMUs, wrt=nimble.neural.WRT_POSITION)
            
            if position_target is not None:
                xT.value = np.array(subsets_joints).reshape(-1)
                x.value = self._nimble.getJointWorldPositions(self.joints)[mask.reshape(-1)]
                Jp.value = self._nimble.getJointWorldPositionsJacobianWrtJointPositions(self.joints)[mask.reshape(-1)]
            

            prob.solve(solver=cp.OSQP, warm_start=True)

            dq_cumulative.value = np.array(dq_cumulative.value) + np.array(dq.value)

            self._nimble.setPositions(q.value + dq.value)

            if np.linalg.norm(dq.value.flatten()) < precision:
                break