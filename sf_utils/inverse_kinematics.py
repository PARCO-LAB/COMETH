import cvxpy as cp
import numpy as np
import nimblephysics as nimble

from COMETH import DynamicSkeleton

def estimate_position_from_IMUs_qp(s:DynamicSkeleton,acc_target=None, gyro_target=None, position_target=None, max_iterations=100, precision=0.0001, dt=1/60):
    
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
    
    q_l.value = np.clip(s.q_l,-1000,1000)
    q_u.value = np.clip(s.q_u,-1000,1000)
    dq_l.value = dt*s.qdot_l
    dq_u.value = dt*s.qdot_u

    # No speed limit in case of positional target
    if position_target is not None:
        dq_l.value = dt*1000*s.qdot_l
        dq_u.value = dt*1000*s.qdot_u
    i=0
    
    # Solving the QP iterativly
    while i < max_iterations:
        i+=1
        # Parameters initialization
        q.value = s._nimble.getPositions()

        if acc_target is not None:
            aT.value = acc_target
            a.value = s._nimble.getAccelerometerReadings(s.IMUs)
            Ja.value = s._nimble.getAccelerometerReadingsJacobianWrt(accs=s.IMUs, wrt=nimble.neural.WRT_POSITION)
        
        if gyro_target is not None:
            wT.value = gyro_target
            w.value = s._nimble.getGyroReadings(s.IMUs)
            Jw.value = s._nimble.getGyroReadingsJacobianWrt(gyros=s.IMUs, wrt=nimble.neural.WRT_POSITION)
        
        if position_target is not None:
            xT.value = np.array(subsets_joints).reshape(-1)
            x.value = s._nimble.getJointWorldPositions(s.joints)[mask.reshape(-1)]
            Jp.value = s._nimble.getJointWorldPositionsJacobianWrtJointPositions(s.joints)[mask.reshape(-1)]
        

        prob.solve(solver=cp.OSQP, warm_start=True)

        dq_cumulative.value = np.array(dq_cumulative.value) + np.array(dq.value)

        s._nimble.setPositions(q.value + dq.value)

        if np.linalg.norm(dq.value.flatten()) < precision:
            break