import numpy as np
import nimblephysics as nimble

def mdh_transform(theta, a, alpha, d):
    """ 
     Compute modified Denavit-Hartenberg T matrix
    """
    cz, sz = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    T = np.array([
        [cz, -sz*ca,  sz*sa, a*cz + d*sz*sa],
        [sz,  cz*ca, -cz*sa, a*sz - d*cz*sa],
        [0,      sa,     ca,     d*ca],
        [0,      0,      0,     1]
    ])
    return T

# One arm kinematics chain model
class ArmModel:
    def __init__(self, dh_params, R_seg_to_imu=None, p_seg_to_imu=None,
                 R_seg_to_marker=None, p_seg_to_marker=None,
                 R_imu_to_cam=None, p_imu_to_cam=None,
                 gravity=np.array([0, 0, -9.80665])): # NOTE: this default parameter have to be consistent with others
        """
        dh_params: list of dict of len equal the DoF
        R_seg_to_imu: list of IMUs rotation with respect to respective segment (order-wise)
        p_seg_to_imu: list of IMUs offset with respect to respective segment (order-wise)
        R_imu_to_cam: rotation matrix R of camera respect to IMU
        p_imu_to_cam: offset array from 
        """
        self.dh = dh_params
        self.R_seg_to_imu = R_seg_to_imu #from imu to segment
        self.p_seg_to_imu = p_seg_to_imu #segment ref sistem
        self.R_seg_to_marker = R_seg_to_marker #from marker to segment
        self.p_seg_to_marker = p_seg_to_marker #segment ref sistem
        self.R_imu_to_cam = R_imu_to_cam
        self.p_imu_to_cam = p_imu_to_cam
        self.g = gravity
        self.n = len(dh_params)  # Total number of DoF

    def forward_kinematics(self, q):
        """Compute positions, rotations and z axes for each joint"""
        T = np.eye(4)
        positions, rotations, z_axes = [], [], []
        for i in range(len(q)):
            link = self.dh[i]
            Ti = mdh_transform(q[i], link['a'], link['alpha'], link['d'])
            T = T @ Ti
            positions.append(np.array(T[:3, 3]))
            rotations.append(T[:3, :3])
            z_axes.append(T[:3, 2])
        return positions, rotations, z_axes
    
    def jacobian_position(self, q):
        """
        Positional jacobian (3*N,N) with respect to the N theta of the kinematics chain.
        """
        positions, _, z_axes = self.forward_kinematics(q)  # p_i, R_i, z_i
        n = len(q)
        k = n - 1
        if not (0 <= k < n):
            raise ValueError(f"to_frame must be in [0, {n-1}]")

        J_all = np.zeros((3*n, n))
        p_list = [np.zeros(3)] + positions
        z_list = [np.array([0.0, 0.0, 1.0])] + z_axes

        for k in range(n):
            pk = positions[k]
            block = np.zeros((3, n))
            for i in range(n):
                if i > k:
                    continue
                zi = z_list[i]
                pi = p_list[i]
                block[:, i] = np.cross(zi, (pk - pi))
            J_all[3*k : 3*(k+1), :] = block

        return J_all

    def inverse_kinematics(self, target, initial_guess=None, max_iterations=100, precision=0.0001):
        """ 
         Compute IK of the arm model throught least square error method
        """
        old_loss = np.inf
        if initial_guess is None:
            q = np.zeros(self.n)
        else:
            q = initial_guess

        for _ in range(max_iterations):
            i=0
            while i<max_iterations:
                pos, _, _ = self.forward_kinematics(q)
                pos = np.array(pos)
                d_loss_d_pos = 2*(pos-target).reshape((1,-1))
                d_pos_d_theta = self.jacobian_position(q) 
                d_loss_d_theta = d_loss_d_pos@d_pos_d_theta
                q -= 0.01 * d_loss_d_theta.reshape(-1)
                i+=1
            pos, _, _ = self.forward_kinematics(q)
            error = (np.array(pos) - target).reshape(-1)
            loss = np.inner(error, error)
            
            if np.abs(old_loss - loss) < precision:
                return q
            old_loss = loss

        return q
    
    def imu_measurement(self, q, dq, ddq, seg_index):
        """
        IMU measure (gyro and acc.) of seg_index segment.
        """
        positions, rotations, z_axes = self.forward_kinematics(q)
        s = seg_index

        p = [positions[0].copy()]
        for i in range(1, s + 1):
            p.append(positions[i] - positions[i - 1])

        omega = np.zeros(3)
        alpha = np.zeros(3)
        a_O   = np.zeros(3)    # origin acceleration

        for i in range(s + 1):
            if i > 0:
                a_O = a_O + np.cross(alpha, p[i]) + np.cross(omega, np.cross(omega, p[i]))
            z = z_axes[i]
            alpha = alpha + z * ddq[i] + np.cross(omega, z) * dq[i]
            omega = omega + z * dq[i]

        # Offset and IMU rotation
        Rw = rotations[s]                            # seg -> world
        r_i_world = Rw @ self.p_seg_to_imu[s]        # offset IMU in world
        R_imu_to_world = Rw @ self.R_seg_to_imu[s]   # imu -> world
        R_world_to_imu = R_imu_to_world.T            # world -> imu

        # IMU acceleration in world
        a_IMU_world = a_O + np.cross(alpha, r_i_world) + np.cross(omega, np.cross(omega, r_i_world))

        gyro  = R_world_to_imu @ omega
        accel = R_world_to_imu @ (a_IMU_world - self.g)  # specific force (accel - g) (minus must be consistent with the self.g sign)

        return np.hstack([gyro, accel])
    
    def new_marker_measurement(self, q):
        """ 
        Wrist marker rotation and position
        """
        positions, rotations, _ = self.forward_kinematics(q)
        pos_m_w = positions[-1]
        R_m_w = rotations[-1] @ self.R_seg_to_imu[7] @self.R_seg_to_marker #marker --> world
        return R_m_w, pos_m_w

    def qr_measurement(self, q):
        """
        World rotation in camera frame.
        """
        _, rotations, _ = self.forward_kinematics(q)
        
        R_cam = rotations[2]@self.R_seg_to_imu[2]@self.R_imu_to_cam # cam --> world
        R_C_marker = R_cam.T # world --> cam
        return R_C_marker
    
def compute_modified_dh(T_list, right : bool = True):
    """
    Compute modified DH parameters for a chain.
    T_list: list of (4,4) matrices joint -> world.
    """
    
    offset = []
    for i in range(len(T_list) - 1):
        T_rel = np.linalg.inv(T_list[i]) @ T_list[i+1] #  i+1 -> i
        dist = np.linalg.norm(T_rel[:3,3])
        offset.append(dist)

    dh_params = [
        #thorax yaw
        {'a': 0.0,  'alpha': np.pi/2, 'd': 0.0},
        #thorax pitch
        {'a': 0.0,  'alpha': -np.pi/2, 'd': 0.0},
        #thorax roll
        {'a': 0.0,  'alpha': 0.0,    'd': offset[0]}, #thorax-shoulder offset
        #shoulder yaw
        {'a': 0,  'alpha': np.pi/2, 'd': 0.0},
        #shoulder pitch
        {'a': 0.0,  'alpha': -np.pi/2, 'd': 0.0},
        #shoulder roll
        {'a':  offset[1], 'alpha': 0, 'd': 0}, #shoulder-elbow offset
        #elbow flex
        {'a': 0.0, 'alpha': -np.pi/2, 'd': 0.0}, #elbow-wrist offset
        #elbow pronation
        {'a':  0.0,  'alpha': 0.0, 'd': offset[2]},
    ]
    return dh_params, offset

def compute_list(s):
    """ 
     Lists of left and right IMUs T matrices
     """
    T_list = []
    # NOTE: This list must be consistent with others part of code.
    for node in ['ulna_r','ulna_l','humerus_r', 'humerus_l', "thorax", "hand_r","hand_l"]:
        body_node: nimble.dynamics.BodyNode = s._nimble.getBodyNode(node)
        T_list.append(body_node.getWorldTransform().matrix())
    T_list_r = [T_list[4],T_list[2],T_list[0], T_list[5]]
    T_list_l = [T_list[4],T_list[3],T_list[1], T_list[6]]

    return T_list_r, T_list_l # body -> world

def compute_world_transf(arm:ArmModel, s, right:bool = True, theta_old = None):
    """
    Compute the global T matrices for each joint of the arm chain. The dh model will be fit into the position of the nimble model throught IK 
    """
    # NOTE: these lists are hardcoded, the output of this function must be consistent.
    if right:
        body_node_list = ["thorax", "thorax","thorax", "humerus_r","humerus_r", "humerus_r","ulna_r", "ulna_r","hand_r"]
    else:
        body_node_list = ["thorax", "thorax","thorax", "humerus_l","humerus_l","humerus_l","ulna_l", "ulna_l", "hand_l"]

    target_positions = []
    # Compute target positions
    for i, node in enumerate(body_node_list):
        body_node: nimble.dynamics.BodyNode = s._nimble.getBodyNode(node)
        T_nimb = body_node.getWorldTransform().matrix()
        if i==0:
            offset = T_nimb[:3,3].reshape(-1)
        else:
            target_positions.append(T_nimb[:3,3].reshape(-1)-offset)

    # Compute IK
    thetas = arm.inverse_kinematics(target=np.array(target_positions), initial_guess=theta_old)
    # Compute T global-->link
    T_global_links = []
    T = np.eye(4)
    for i, params in enumerate(arm.dh):
        T = T @ mdh_transform(thetas[i], params['a'], params['alpha'], params['d'])
        T_global_links.append(np.linalg.inv(T))

    return T_global_links, thetas

def compute_imu_ori(arm:ArmModel, s, T_i_w_list_all, right:bool = True, initial_guess = None):
    """ 
    Compute the IMU -> link list of R for the DH model, given the IMUs world orientation
    """
    T_dh, _ = compute_world_transf(arm, s, right,theta_old=initial_guess) # world -> link

    T_w_link = np.array([T_dh[2], T_dh[5], T_dh[7]]) 

    if right:
        T_i_w_list = np.array([T_i_w_list_all[4], T_i_w_list_all[2], T_i_w_list_all[0]]) # thorax, right_elbow, right_wrist
    else:
        T_i_w_list = np.array([T_i_w_list_all[4], T_i_w_list_all[3], T_i_w_list_all[1]]) # thorax, left_elbow, left_wrist


    T_i_link = T_w_link @ T_i_w_list # imu -> link
    
    return [T_i_link[i, :3, :3].reshape(3,3) for i in range(T_i_link.shape[0])]