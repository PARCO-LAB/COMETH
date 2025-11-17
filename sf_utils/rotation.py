import numpy as np

# All the following operations impling quaternions consider them as scalar-first unless specified
#TODO: many of these functions can be performed with SciPy, check want can be done.
def quat_conj(q: np.array):
    """
    Return the conjugate of the given quaternion.
    ### Args
    - q: quaternion or batch of quaternions with shape (..., 4).
    ### Return
    - The conjugate of the quaternion
    """
    assert q.shape[-1] == 4
    qc = q.copy()
    qc[..., 1:] *= -1
    return qc

def batch_quat_inv(q: np.array):
    """
    Invert a batch of quaternions.
    ## Arg
    - q: array of quaternions with shape (N, 4)

    ## Return
    - array of inverted quaternions with shape (N, 4).
    """
    assert q.ndim == 2 and q.shape[1] == 4
    
    conjugate = q * np.array([1, -1, -1, -1])
    
    norm_sq = np.sum(q**2, axis=1, keepdims=True)
    
    return conjugate / norm_sq

def batch_norm_quat(q):
    """
    Quaternion normalization for a batch of size (B,4)
    """
    assert q.ndim == 2 and q.shape[1] == 4

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    return q / norms

def quat_multiply(q1: np.array, q2: np.array):
    """
    Multiplication of quaternions as q1 x q2.
    ### Args
    - q1: first quaternion of shape (4)
    - q2: second quaternion of shape (4)
    ### Return
    - the quaternion of the multiplication result
    """
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], axis=-1)

def batch_quat_mul(q1: np.array, q2: np.array):
    """
    Muliplication of a batch of quaternions of shape (B,4). q1 x q2
    ## Args
    - q1, q2: batch of quaternions to multiply.
    ## Returns
    - The batch of results
    """

    assert q1.shape == q2.shape
    assert q1.shape[-1] == 4

    w1, x1, y1, z1 = np.split(q1, 4, axis=1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.concatenate([w, x, y, z], axis=1)

def quat_to_rotmat(q: np.array):
    """
    Convert quaternion q = [q0, q1, q2, q3] in DCM R (3x3).
    ### Args
    - q: quaternion to convert.
    ### Return
    - R (np.array): rotation matrix of shape (3,3)
    """

    assert q.shape == (4)
    q0, q1, q2, q3 = q
    R = np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3),         2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),             q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),             2*(q2*q3 + q0*q1),         q0*q0 - q1*q1 - q2*q2 + q3*q3]
    ])
    return R

def batch_quat_to_rotmat(q: np.array):
    """
    Convert a batch of quaternions q (Bx4) into a batch of DCMs R (Bx3x3).
    ### Args
    - q: batch of quaternions to convert of shape (B,4).
    ### Return
    - R (np.array): batch of rotation matrices of shape (B,3,3)
    """
    assert q.shape[-1] == 4

    R = []
    for q_row in q: 
        R.append(quat_to_rotmat(q_row.reshape(4)))

    return np.array(R)

# TODO: merge these two functions without naive for cycle
def rotmat_to_quat(R):
    """ 
    Convert a given DCM matrix into corresponding quaternion
    ## Args
    - R: Rotation matrix of shape (3,3)
    ## Returns
    - The corresponding normalized quaternion of shape (4)
    """

    assert R.shape == (3, 3)

    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * x
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * y
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * z
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)  # Normalize
    return q

def batch_rotmat_to_quat(R):
    """
    Convert a batch of DCM matrices R into a batch of rotation quaternions.
    ## Args
    - R: batch of DCM matrices of shape (B,3,3)
    ## Return
    - Batch of corresponding quaternions of shape (B,4)
    """
    rst = []
    for i in range(len(R)):
        rst.append(rotmat_to_quat(R[i]))
    return np.array(rst)

def rotate_vector_by_quat(v: np.array, q:np.array) -> np.array:
    """
    Rotate batch of vectors v [...,3] with a quaternion q [...,4]
    ### Args
    - v: batch of vector with shape (...,3)
    - q: batch of quaternions of shape (...,4)
    ### Return
    - batch of rotated vectors with shape (...,3)
    """

    v_as_quat = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return quat_multiply(quat_multiply(q, v_as_quat), quat_conj(q))[..., 1:]

#TODO: merge these 2 function in order to avoid for cycle but keep attention to gimbal lock
def rotmat_to_euler(R: np.array):
    """ 
    Convert DCM matrix R into xyz eulerian angles in rad.
    ## Args
    - R: Rotational matrix with shape (3,3)
    ## Return
    - roll, pitch yaw in rad
    """

    assert R.shape == (3, 3)
    # Handle singularities (gimbal lock)
    if abs(R[2, 0]) < 1.0:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: pitch = ±90deg
        pitch = np.pi / 2 if R[2, 0] <= -1.0 else -np.pi / 2
        roll = np.arctan2(-R[0, 1], -R[0, 2])
        yaw = 0.0

    return roll, pitch, yaw

def batch_rotmat_to_euler(R:np.array):
    """
    Convert batch of DCM matrices R into batch of xyz eulerian angles in rad.
    ## Args
    - R: Rotational matrices with shape (B,3,3)
    ## Return
    - Array of shape (B,3). Each column is, respectively, roll, pitch, and yaw. 
    """
    angles = []
    for i in range(len(R)):
        angles.append([rotmat_to_euler(R[i].reshape((3,3)))])
    return np.array(angles)

#TODO: rewrite without naive for cycle
def batch_euler_to_rotmat(angles:np.array):
    """
    Convert a batch of eulerian angles with shape (Bx3) and format [roll, pitch, yaw] into a batch of DCM matrixes.
    ## Args
    - angles: vector of angles with shape (Bx3)
    ## Return
    - Batch of DCM matrixes
    """
    R=[]
    for i in range(len(angles)):
        cr, sr = np.cos(angles[i,0]), np.sin(angles[i,0])
        cp, sp = np.cos(angles[i,1]), np.sin(angles[i,1])
        cy, sy = np.cos(angles[i,2]), np.sin(angles[i,2])
        
        # Rotation matrix from body to world (ZYX)
        R.append(np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,    cp*sr,            cp*cr]
        ]))
    return np.array(R)

# TODO: test the batch behavior of this function
def euler_to_quat(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to quaternion.
    # Args:
    - Eulerian angles roll, pitch, yaw
    # Return:
    - The corresponding rotation quaternion of shape (...,4)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])

#TODO: Make it with batch
def euler_to_gravity(roll, pitch, yaw, g=9.81):
    """ 
    Given a world orientation in euler angles, calculate the gravity force reading of an IMU in the rotated frame.
    ## Args
    - roll, pitch, yaw angles in this order
    - g: the value of gravity in world frame (considerated z up)
    ## Return
    - An array with gravity components
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # Rotation matrix from body to world (ZYX)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,    cp*sr,            cp*cr]
    ])
    
    # Gravity in world frame
    g_world = np.array([0, 0, g])
    
    # Gravity in body frame
    g_body = R.T @ g_world
    return g_body

def apply_T(T: np.array, v: np.array):
    """ 
    Apply rototranslation matrix T (Bx4x4) to the vector v (Bx3). B is batch size.
    ### Args
    - T: batch of rototranslation matrices of shape (B,4,4)
    - v: batch of vectors of shape (Bx3)
    ### Return
    - The rototranslated batch of vectors with shape (Bx3)
    """
    assert T.shape[0] == v.shape[0]
    assert T.shape[1:] == (4, 4)
    assert v.shape[1] == 3
    
    v_h = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)
    v_transformed = np.einsum('bij,bj->bi', T, v_h)
    
    return v_transformed[:, :3].reshape(-1,3)
