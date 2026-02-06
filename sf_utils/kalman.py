import numpy as np
from sf_utils.rotation import so3_log, skew

class DualArmEKF:
    def __init__(self, arms, Q, R_dict, initial_state = None):
        """
        arms: list of ArmModel [right, left]
        Q: process noise matrix
        R_dict: dict of measures noises
        """
        
        self.arms = arms
        self.n_arm = arms[0].n # n dof
        self.P = np.eye(6 * self.n_arm) * 1e-4
        self.Q = Q
        self.R = R_dict

        if initial_state is not None:
            assert (initial_state.shape == (6*self.n_arm,))
            self.state = initial_state
        else:
            self.state = np.zeros(6 * self.n_arm)  # 48 variables
        
    def predict(self, dt):
        """
        State prediction with Uniformly Accelerated Motion model
        """
        n = self.n_arm
        # Right arm
        qR = self.state[0:n]
        qdR = self.state[n:2*n]
        qddR = self.state[2*n:3*n]
        qR = qR + qdR*dt + 0.5*qddR*dt*dt
        qdR = qdR + qddR*dt
        # Left arm
        qL = self.state[3*n:4*n]
        qdL = self.state[4*n:5*n]
        qddL = self.state[5*n:6*n]
        qL = qL + qdL*dt + 0.5*qddL*dt*dt
        qdL = qdL + qddL*dt
        # State update
        self.state = np.hstack([qR, qdR, qddR, qL, qdL, qddL])
        # Covariance update
        A = np.eye(6*n)
        for offset in [0, 3*n]:
            A[offset:offset+n, offset+n:offset+2*n] = np.eye(n)*dt
            A[offset:offset+n, offset+2*n:offset+3*n] = np.eye(n)*0.5*dt*dt
            A[offset+n:offset+2*n, offset+2*n:offset+3*n] = np.eye(n)*dt
        self.P = A @ self.P @ A.T + self.Q
        
    def predict_no_acc(self, dt):
        """
        State prediction with no acceleration model
        """
        n = self.n_arm
        # Right arm
        qR = self.state[0:n]
        qdR = self.state[n:2*n]
        qddR = self.state[2*n:3*n]
        # Left arm
        qL = self.state[3*n:4*n]
        qdL = self.state[4*n:5*n]
        qddL = self.state[5*n:6*n]
        # State update
        self.state = np.hstack([qR, qdR, qddR, qL, qdL, qddL])
        # Covariance update
        A = np.eye(6*n)
        for offset in [0, 3*n]:
            A[offset:offset+n, offset+n:offset+2*n] = np.eye(n)*dt
            A[offset:offset+n, offset+2*n:offset+3*n] = np.eye(n)*0.5*dt*dt
            A[offset+n:offset+2*n, offset+2*n:offset+3*n] = np.eye(n)*dt
        self.P = A @ self.P @ A.T + self.Q

    def update_imu(self, z_all, R_block=None, use_joseph=True):
        """
        Update with IMU mesuremeant. 
        
        Parametri
        ---------
        z_all : iterable of tuple (arm_id, seg_index, z)
            - arm_id: 0 = right, 1 = left
            - seg_index: segment index in the ArmModel chain
            - z: IMU measure as array (6,) = [wx, wy, wz, fx, fy, fz]
        R_block : np.ndarray o None
            - None: use block-diagonal with self.R['imu'] for each block (6x6)
            - If given: noises Covariance
        use_joseph : bool
            - If True: use Joseph formula.
        """
        n = self.n_arm
        x_dim = len(self.state)

        z_stack = []
        h_stack = []
        H_rows = []

        blocks_count = 0

        for (arm_id, seg_index, z) in z_all:
            # State offset for the specific arm
            idx_start = arm_id * 3 * n
            arm = self.arms[arm_id]

            # Arm state (q, qd, qdd)
            x_local = self.state[idx_start:idx_start + 3*n]
            q   = x_local[0:n]
            qd  = x_local[n:2*n]
            qdd = x_local[2*n:3*n]

            h_i = arm.imu_measurement(q, qd, qdd, seg_index)     # (6,)
            
            H_local = self._numeric_jacobian(func=lambda xl: arm.imu_measurement(xl[:n], xl[n:2*n], xl[2*n:], seg_index), active_idx = None, x=x_local)  # (6, 3*n)

            z_stack.append(z.reshape(6))
            h_stack.append(h_i.reshape(6))

            # Build one row-block of the global jacobian (6 x x_dim)
            H_i = np.zeros((6, x_dim))
            H_i[:, idx_start:idx_start + 3*n] = H_local
            H_rows.append(H_i)

            blocks_count += 1

        if blocks_count == 0:
            return

        # Final stack
        z_stack = np.vstack(z_stack)             # (6M x 1)
        h_stack = np.vstack(h_stack)             # (6M x 1)
        H = np.vstack(H_rows)                    # (6M x x_dim)
        y = (z_stack - h_stack).reshape(-1)      

        if R_block is None:
            Rm_single = self.R['imu']            # (6 x 6)
            Rm = self._block_diag(Rm_single, blocks_count)  # (6M x 6M)
        else:
            Rm = R_block

        # Innovation covariance and gain
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ y

        # Covariance update
        I = np.eye(x_dim)
        if use_joseph:
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T
        else:
            self.P = (I - K @ H) @ self.P

    def update_marker(self, arm_id, z_pos, z_rot, max_iters = 10):
        """ 
         Updating with marker data using iekf
        """
        def block_diag(*arrs):
            shapes = np.array([a.shape for a in arrs])
            out_shape = np.sum(shapes, axis=0)
            out = np.zeros(out_shape, dtype=arrs[0].dtype)

            r, c = 0, 0
            for a in arrs:
                rows, cols = a.shape
                out[r:r+rows, c:c+cols] = a
                r += rows
                c += cols
            return out

        n = self.n_arm
        idx_start = arm_id * 3 * n
        arm = self.arms[arm_id]
        q = self.state[idx_start:idx_start+n].copy()
        P = self.P.copy()

        R = block_diag(self.R['marker_pos'], self.R['marker_rot'])

        for k in range(max_iters):


            R_pred, p_pred = arm.new_marker_measurement(q)

            r_p = z_pos - p_pred
            r_R = so3_log(z_rot @ R_pred.T)
            r = np.hstack([r_p, r_R])

            Hp_local = self._marker_pos_jacobian(q, arm)
            Hp = np.zeros((3, len(self.state)))
            Hp[:, idx_start:idx_start+n] = Hp_local

            Hr_local = self._rotation_jacobian(q, z_rot, arm)
            Hr = np.zeros((3, len(self.state)))
            Hr[:, idx_start:idx_start+n] = Hr_local
            H = np.vstack([Hp, Hr])
            S = H @ P @ H.T + R
            
            K = P @ H.T @ np.linalg.inv(S)

            dq = K @ r
            q_new = q + dq[idx_start:idx_start+n]

            if np.linalg.norm(dq) < 1e-6 or k==(max_iters-1):
                q = q_new
                break
            
            q = q_new
            I = np.eye(len(self.state))
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

        self.state[idx_start:idx_start+n] = q
        self.P = P

    def update_qr(self, arm_id, z_rot):
        n = self.n_arm
        idx_start = arm_id * 3 * n
        arm = self.arms[arm_id]
        q = self.state[idx_start:idx_start+n]
        R_pred = arm.qr_measurement(q) # seg -> cam
        r = so3_log(z_rot @ R_pred.T)

        Hr_local= self._rotation_jacobian(q, z_rot, arm, idx = 3)
        Hr = np.zeros((3, len(self.state)))
        Hr[:, idx_start:idx_start+n] = Hr_local
        Sr = Hr @ self.P @ Hr.T + self.R['qr_rot']
        Kr = self.P @ Hr.T @ np.linalg.inv(Sr)
        self.state += Kr @ r
        I = np.eye(len(self.state))
        self.P = (I - Kr @ Hr) @ self.P @ (I - Kr @ Hr).T + Kr @ self.R['qr_rot'] @ Kr.T

    def _numeric_jacobian(self, func, x, active_idx=None, eps=None):
        """
        Compute numeric jacobian of the 'func' function with respect to vector x.
        - active_idx: array/list of index to derivate. If None, all index considerated. Non-active columns are keep to zero.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n = x.size

        # Compute y and normalize
        y0 = func(x)
        y0 = np.atleast_1d(np.asarray(y0, dtype=np.float64)).ravel()
        m = y0.size

        J = np.zeros((m, n), dtype=np.float64)

        # Active index
        if active_idx is None:
            active_idx = np.arange(n, dtype=int)
        else:
            active_idx = np.asarray(active_idx, dtype=int).ravel()
            
            if active_idx.size == 0:
                return J
            if np.any((active_idx < 0) | (active_idx >= n)):
                raise IndexError("active_idx with out of range indexes")
            
            _, unique_pos = np.unique(active_idx, return_index=True)
            active_idx = active_idx[np.sort(unique_pos)]

        if eps is None:
            eps_vec = np.sqrt(np.finfo(np.float64).eps) * (1.0 + np.abs(x))
        else:
            eps_vec = np.asarray(eps, dtype=np.float64)
            if eps_vec.size == 1:
                eps_vec = np.full(n, float(eps_vec), dtype=np.float64)
            elif eps_vec.size != n:
                raise ValueError("eps must be a scalar or an array with len n")
        
        for j in active_idx:
            h = eps_vec[j]
            x_plus  = x.copy(); x_plus[j]  += h
            x_minus = x.copy(); x_minus[j] -= h
            y_plus  = func(x_plus)
            y_minus = func(x_minus)
            y_plus  = np.atleast_1d(np.asarray(y_plus,  dtype=np.float64)).ravel()
            y_minus = np.atleast_1d(np.asarray(y_minus, dtype=np.float64)).ravel()
            
            if not (np.all(np.isfinite(y_plus)) and np.all(np.isfinite(y_minus))):
                raise FloatingPointError(f"f(x) not finite for index {j} (h={h})")

            J[:, j] = (y_plus - y_minus) / (2.0 * h)

        return J
    
    def _marker_pos_jacobian(self, q, arm):
        """
        Marker position Jacobian (3*N,N) with respect to N theta.
        """
        positions, _ , z_axes = arm.forward_kinematics(q)  # p_i, R_i, z_i
        n = len(q)
        k = n - 1
        if not (0 <= k < n):
            raise ValueError(f"to_frame deve essere in [0, {n-1}]")

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

        return J_all[3*(n-1):3*n, :]

    def _so3_right_jacobian_inverse(self, phi, eps=1e-12):
        """
        Inverse of so3 jacobian.
        Small changes aprox for formula:
        J_r^{-1}(phi) circa I - 1/2[phi]_x + 1/12[phi]_x^2  (O(||phi||^4))
        """
        theta = np.linalg.norm(phi)
        I = np.eye(3)
        S = skew(phi)
        if theta < eps:
            return I - 0.5 * S + (1.0/12.0) * (S @ S)

        theta2 = theta * theta
        c = (1.0/theta2) * (1.0 - (theta*np.sin(theta)) / (2.0*(1.0 - np.cos(theta))))
        return I - 0.5 * S + c * (S @ S)

    def _rotation_jacobian(self, q, R_meas, arm, idx = None):
        """
        Stack (3*N, N) of jacobian r_k(q) = so3_log(R_meas_k @ R_k(q)^T)
        """

        _, rotations, _ = arm.forward_kinematics(q)
        n = len(q)
        J_all = np.zeros((3*n, n))
        ez = np.array([0.0, 0.0, 1.0])

        if idx is None:
            idx = n
            R_mark = arm.R_seg_to_marker
        else:
            R_mark = arm.R_imu_to_cam 

        for k in range(n):
            R_pred = rotations[k]@arm.R_seg_to_imu[k]@R_mark
            if idx == 3:
                R_pred = R_pred.T
            if R_meas is None:
                R_meas = np.eye(3)
            R_err  = R_meas @ R_pred.T
            r      = so3_log(R_err)
            Jr_inv = self._so3_right_jacobian_inverse(r)

            Jw = np.zeros((3, n))
            for i in range(k + 1):
                Jw[:, i] = rotations[i] @ ez

            J_block = - Jr_inv @ Jw
            J_all[3*k : 3*(k+1), :] = J_block

        return J_all[3*(idx-1):3*idx, :]

    def _block_diag(self, R_single, count):
        """
       Create a block-diagonal matrix with 'count' R_single (k x k) matrices.
        Final shape: (count*k) x (count*k)
        """
        k = R_single.shape[0]
        R = np.zeros((count*k, count*k))
        for i in range(count):
            R[i*k:(i+1)*k, i*k:(i+1)*k] = R_single
        return R
    