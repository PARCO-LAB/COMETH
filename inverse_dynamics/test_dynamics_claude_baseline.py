import pandas as pd
from COMETH import Skeleton,DynamicSkeleton
import nimblephysics as nimble
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Read ground truth data from CSV
markers = pd.read_csv("/home/emartini/COMETH/inverse_dynamics/data.csv")
s12 = Skeleton()
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
target = []
for i in range(markers.shape[0]):
    row = []
    for kp in markers_dict.keys():
        p = np.array([markers[markers_dict[kp]+"_x"][i],markers[markers_dict[kp]+"_y"][i],markers[markers_dict[kp]+"_z"][i]])
        # rotate the 3d point -90 on the x axis (from y up to z up)
        p_n = Rz.dot(Rx.dot(p))
        # print(p,p_n)
        row += p_n.tolist()
    target.append(row)
target = np.array(target)


import numpy as np
import cvxpy as cp
from scipy.sparse.linalg import ArpackError

def diagnose_matrix(name, A, to_print=True):
    U, sv, Vt = np.linalg.svd(A, full_matrices=False)
    if to_print:
        print(f"{name}: shape={A.shape}, σ_max={sv[0]:.2e}, σ_min={sv[-1]:.2e}, "
            f"rank={np.sum(sv > 1e-10)}/{min(A.shape)}, cond={sv[0]/sv[-1]:.2e}")
    return sv


def get_world_contact_points(contact_info):
    """
    Transform contact points for ground constraints definition and plotting.
    """
    world_points = []
    
    for body_node, local_offset in contact_info:
        offset_np = np.array(local_offset).flatten()
        T_world = body_node.getWorldTransform().matrix()
        offset_homo = np.append(offset_np, 1.0)
        world_pos_homo = T_world @ offset_homo
        world_pos = world_pos_homo[:3]
        world_points.append(world_pos)

    return world_points


def estimate_contact_points(skeleton, target = None):
    left_calcn = skeleton._nimble.getBodyNode("calcn_l")
    right_calcn = skeleton._nimble.getBodyNode("calcn_r")

    # More refined contact point positions - adjust for better foot-ground contact detection
    left_calc_offset_L = [0.0,  0.0, -0.03]  # Heel (slightly forward)
    left_calc_offset_R = [0.0,  0.0, 0.03]   # Heel (slightly forward)
    right_calc_offset_L = [0.0,  0.0, -0.03]  # Heel (slightly forward)
    right_calc_offset_R = [0.0,  0.0, 0.03]   # Heel (slightly forward)

    left_toes_offset_L = [0.15,  0.0, -0.03]  # Toe (more forward and slightly up)
    left_toes_offset_R = [0.15,  0.0, 0.03]   # Toe (more forward and slightly up)
    right_toes_offset_L = [0.15,  0.0, -0.03]  # Toe (more forward and slightly up)
    right_toes_offset_R = [0.15,  0.0, 0.03]   # Toe (more forward and slightly up)

    contact_info = [
            (left_calcn, left_calc_offset_L),
            (left_calcn, left_calc_offset_R),
            (right_calcn, right_calc_offset_L),
            (right_calcn, right_calc_offset_R),
            (left_calcn, left_toes_offset_L),
            (left_calcn, left_toes_offset_R),
            (right_calcn, right_toes_offset_L),
            (right_calcn, right_toes_offset_R),
    ]

    res_world = get_world_contact_points(contact_info)
    to_keep = []

    # Use a more robust threshold and consider the z-coordinate more carefully
    for i, c_point in enumerate(contact_info):
        # Check if point is close enough to ground (0.03 instead of 0.04 for better detection)
        z_coord = res_world[i][-1]
        if z_coord < 0.03:  # More lenient threshold
            to_keep.append(c_point)
        # Also include points that are very close to ground but slightly above (for stability)
        elif z_coord < 0.05 and z_coord > 0.0:
            to_keep.append(c_point)

    return to_keep

def estimate_contact_points_old(skeleton, target = None):
    left_calcn = skeleton._nimble.getBodyNode("calcn_l")
    right_calcn = skeleton._nimble.getBodyNode("calcn_r")

    # Use more conservative positions for better stability
    left_calc_offset_L = [0.0,  0.0, -0.03]  # Heel (slightly forward)
    left_calc_offset_R = [0.0,  0.0, 0.03]   # Heel (slightly forward)
    right_calc_offset_L = [0.0,  0.0, -0.03]  # Heel (slightly forward)
    right_calc_offset_R = [0.0,  0.0, 0.03]   # Heel (slightly forward)

    left_toes_offset_L = [0.15,  0.0, -0.03]  # Toe (more forward and slightly up)
    left_toes_offset_R = [0.15,  0.0, 0.03]   # Toe (more forward and slightly up)
    right_toes_offset_L = [0.15,  0.0, -0.03]  # Toe (more forward and slightly up)
    right_toes_offset_R = [0.15,  0.0, 0.03]   # Toe (more forward and slightly up)

    if target is None:
        contact_info = [
            (left_calcn, left_calc_offset_L),
            (left_calcn, left_calc_offset_R),
            (right_calcn, right_calc_offset_L),
            (right_calcn, right_calc_offset_R),
            (left_calcn, left_toes_offset_L),
            (left_calcn, left_toes_offset_R),
            (right_calcn, right_toes_offset_L),
            (right_calcn, right_toes_offset_R),
        ]
    else:
        lankle_z = target[35]
        rankle_z = target[32]
        ground = 0.1
        # More robust contact detection - if both feet are off ground, use default
        if lankle_z > ground and rankle_z < ground:
            # Right foot on ground
            contact_info = [
                (right_calcn, right_calc_offset_L),
                (right_calcn, right_calc_offset_R),
                (right_calcn, right_toes_offset_L),
                (right_calcn, right_toes_offset_R),
            ]
        elif rankle_z > ground and lankle_z < ground:
            # Left foot on ground
            contact_info = [
                (left_calcn, left_calc_offset_L),
                (left_calcn, left_calc_offset_R),
                (left_calcn, left_toes_offset_L),
                (left_calcn, left_toes_offset_R),
            ]
        else:
            # Both feet or neither on ground - use default contact points
            contact_info = [
                (left_calcn, left_calc_offset_L),
                (left_calcn, left_calc_offset_R),
                (right_calcn, right_calc_offset_L),
                (right_calcn, right_calc_offset_R),
                (left_calcn, left_toes_offset_L),
                (left_calcn, left_toes_offset_R),
                (right_calcn, right_toes_offset_L),
                (right_calcn, right_toes_offset_R),
            ]
    return contact_info


def get_contact_jacobian_deriv_times_dq(skeleton, contact_info, dq):
    """
    Calcola il termine di accelerazione di bias (dJ_c * dq) per i punti di contatto.        
    Returns:
        dJc_dq: Vettore numpy 1D (24,) contenente le accelerazioni di bias (xyzxyz...)
    """
    
    # IMPORTANTE: Affinché Nimble calcoli la derivata temporale dello Jacobiano (che 
    # dipende da q e da dq), il motore fisico DEVE essere aggiornato allo stato corrente.
    # Assumiamo che skeleton.setPositions(q) sia già stato chiamato nel main loop,
    # ma per sicurezza ribadiamo le velocità:
    # skeleton.setVelocities(dq)
    
    num_contacts = len(contact_info)
    dJc_dq = np.zeros(num_contacts * 3)
    
    for i, (body_node, offset) in enumerate(contact_info):       
        # 1. Estrai la derivata temporale dello Jacobiano Lineare (Matrice 3 x 49)
        # Nimble calcola analiticamente dJ/dt considerando le velocità attuali
        dJ_i = skeleton._nimble.getLinearJacobianDeriv(body_node,offset)
        
        # 2. Moltiplica la matrice per le velocità articolari per ottenere l'accelerazione 3D
        bias_accel_i = dJ_i @ dq
        
        # 3. Inserisci l'accelerazione (x, y, z) nel vettore piatto finale
        idx_start = i * 3
        idx_end = idx_start + 3
        dJc_dq[idx_start:idx_end] = bias_accel_i
        
    return dJc_dq

def get_task_jacobian_derivative_times_dq(skeleton, keypoint_joints, dq):
    """
    Calculate convettive acceleration J_dot * q_dot for each keypoint.
    """
    dJ_dq_list = []
    
    for joint in keypoint_joints:
        body_node = joint.getChildBodyNode() 
        # Get the offset
        T_child = joint.getTransformFromChildBodyNode()
        local_offset = T_child.translation() # numpy array [3, 1] o [3,]
        
        # Extract the jacobian
        J_dot_linear = skeleton._nimble.getLinearJacobianDeriv(body_node, local_offset)
        
        # Calculate convettive acceleration
        acc_conv = J_dot_linear @ dq 
        
        dJ_dq_list.append(acc_conv)
        
    dJ_kp_dq = np.concatenate(dJ_dq_list)
    
    return dJ_kp_dq

def get_contact_jacobian(skeleton, contact_info):
    """
    Calculate contact jacobian.
    """
    J_c_list = []
    
    for body_node, local_offset in contact_info:
        offset_np = np.array(local_offset, dtype=np.float64).reshape(3, 1)
        J_point = skeleton._nimble.getLinearJacobian(body_node, offset_np)
        J_c_list.append(J_point)
        
    J_c = np.vstack(J_c_list)
    
    return J_c

def qpid(skeleton, x_t, dt=0.033, mu=0.8, excluded_DOFs=None):
    # Protezione per lista mutabile come argomento di default
    if excluded_DOFs is None:
        excluded_DOFs = [
            11, 12,18,19,   # subtalar, mtp
            37,38,47,48,    # wrist
            26,27,28,       # head
            36,46,          # elbow pronation
            ]

    n_dof_full = skeleton._nimble.getNumDofs()   # TOTAL Degrees of Freedom (es. 49)
    n_act_full = n_dof_full - 6                  # Actuated Degrees of Freedom (es. 43)
    n_kp = 12
    
    # --- 1. LOGICA DI ESCLUSIONE (MASKING) ---
    # Trova tutti i DOFs "attivi"
    active_dofs = [i for i in range(n_dof_full) if i not in excluded_DOFs]
    n_dof = len(active_dofs)  # Nuova dimensione del problema

    # Debug: Check if we have valid dimensions
    if n_dof <= 0:
        print("Errore: Nessun DOF attivo trovato")
        return None
    
    # Trova i DOFs attivi che sono anche attuati (indice >= 6)
    active_act_dofs = [d for d in active_dofs if d >= 6]
    n_act = len(active_act_dofs)
    
    # --- 2. ESTRAZIONE STATO COMPLETO ---
    q = skeleton._nimble.getPositions()
    dq = skeleton._nimble.getVelocities()
    
    # Estraiamo le matrici FULL size prima di tagliarle
    M_full = skeleton._nimble.getMassMatrix() + np.eye(n_dof_full) * 1e-6
    H_cg_full = skeleton._nimble.getCoriolisAndGravityForces()
    
    J_kp_full = skeleton._nimble.getJointWorldPositionsJacobianWrtJointPositions(skeleton.joints)
    dJ_kp_dq = get_task_jacobian_derivative_times_dq(skeleton, skeleton.joints, dq) # Resta FULL (task space)
    x_current = np.array(skeleton._nimble.getJointWorldPositions(skeleton.joints))

    contact_info = estimate_contact_points(skeleton,target=x_t)
    n_contacts = len(contact_info)
    J_c_full = get_contact_jacobian(skeleton, contact_info) 

    # --- 3. SLICING DELLE MATRICI ---
    # Tagliamo via le righe/colonne dei DOFs ignorati usando np.ix_ o lo slicing classico
    M = M_full[np.ix_(active_dofs, active_dofs)]
    H_cg = H_cg_full[active_dofs]
    J_kp = J_kp_full[:, active_dofs]
    J_c = J_c_full[:, active_dofs]

    # PD-controller
    Kp = 150.0 
    Kd = 20.0  
    x_t_flat = np.array(x_t).flatten()
    # x_dot usa J_kp_full perché dq contiene le velocità correnti (anche dei giunti che stiamo per "congelare")
    x_dot_current = J_kp_full @ dq 
    x_des_ddot = Kp * (x_t_flat - x_current) - Kd * x_dot_current

    # --- 4. VARIABILI CVXPY (DIMENSIONI RIDOTTE) ---
    ddq = cp.Variable(n_dof)           # Ora è più piccolo (es. 47)
    tau = cp.Variable(n_act)           # Ora è più piccolo (es. 41)
    fc = cp.Variable(n_contacts * 3)   
    delta = cp.Variable(n_kp * 3, nonneg=True)      
    tau_virtual = cp.Variable(6) 

    # --- 5. MATRICI DI SELEZIONE DINAMICHE ---
    # Mappa le coppie attive (tau) all'equazione dinamica (che ha dimensione n_dof)
    S_T = np.zeros((n_dof, n_act))
    act_idx = 0
    for i, dof in enumerate(active_dofs):
        if dof >= 6:
            S_T[i, act_idx] = 1.0
            act_idx += 1
            
    # Mappa le forze virtuali del bacino (se il bacino non è mai escluso) all'equazione
    S_vb = np.zeros((n_dof, 6))
    for i, dof in enumerate(active_dofs):
        if dof < 6:
            S_vb[i, dof] = 1.0
            
    # Matrice Attrito (invariata)
    V = np.zeros((n_contacts * 5, n_contacts * 3))
    for i in range(n_contacts):
        row_idx, col_idx = i * 5, i * 3
        V[row_idx:row_idx+5, col_idx:col_idx+3] = np.array([
            [-1,  0,  mu], [ 1,  0,  mu], [ 0, -1,  mu], [ 0,  1,  mu], [ 0,  0,   1]
        ])
    
    # --- 6. GESTIONE LIMITI (Calcolati Full, poi tagliati) ---
    q_L, q_U = skeleton.q_l, skeleton.q_u
    dq_L, dq_U = skeleton.qdot_l * 2, skeleton.qdot_u * 2
    gamma, dt_sq = 0.5, dt**2

    ddq_max_pos = gamma * (q_U - q - (dq * dt)) / dt_sq
    ddq_min_pos = gamma * (q_L - q - (dq * dt)) / dt_sq
    ddq_max_vel = (dq_U - dq) / dt
    ddq_min_vel = (dq_L - dq) / dt

    ddq_ub_full = np.minimum(ddq_max_pos, ddq_max_vel)
    ddq_lb_full = np.maximum(ddq_min_pos, ddq_min_vel)

    infeasible_mask = ddq_lb_full > ddq_ub_full
    if np.any(infeasible_mask):
        mid = 0.5 * (ddq_lb_full + ddq_ub_full)
        ddq_lb_full = np.where(infeasible_mask, mid - 1e-3, ddq_lb_full)
        ddq_ub_full = np.where(infeasible_mask, mid + 1e-3, ddq_ub_full)

    # Taglia i limiti per i DOFs attivi
    ddq_ub = ddq_ub_full[active_dofs]
    ddq_lb = ddq_lb_full[active_dofs]

    # --- 7. Z-CONTACTS ---
    J_c_z_full = J_c_full[2::3, :]
    J_c_z = J_c_z_full[:, active_dofs] # Slice
    
    contact_pts = get_world_contact_points(contact_info)
    z_current = np.array([p[2] for p in contact_pts]) 

    dJ_c_dq_z_full = get_contact_jacobian_deriv_times_dq(skeleton, contact_info, dq)
    dJ_c_dq_z = dJ_c_dq_z_full[2::3]  

    z_ddot = J_c_z @ ddq + dJ_c_dq_z
    
    # v_current_z calcolato sulla posa attuale (Full)
    v_current_z = J_c_z_full @ dq
    z_next = z_current + (v_current_z * dt) + (z_ddot * (dt**2))

    # --- 8. VINCOLI CVXPY ---
    constraints = [
        M @ ddq + H_cg == S_T @ tau + J_c.T @ fc + S_vb @ tau_virtual,
        J_kp @ ddq + dJ_kp_dq == x_des_ddot + delta,
        V @ fc >= 0,
        tau <= 100.0, # Avendo creato tau solo per i dof attuati, non serve moltiplicare per S_T qui!
        tau >= -100.0,
        ddq <= ddq_ub,
        ddq >= ddq_lb,
        # z_next + delta_z >= 0
        z_next >= 0,
    ]
    
    # --- 9. PESI E REGOLARIZZAZIONE ---
    sv_Jkp = diagnose_matrix("J_kp", J_kp, to_print=False)
    null_dof_indices = np.where(sv_Jkp < 1e-6)[0]

    # Add more robust regularization to handle ill-conditioned matrices
    W_ddq = np.eye(n_dof) * 1.0
    W_ddq[null_dof_indices, null_dof_indices] = 100.0  # Increased regularization for null space

    W_tau = np.eye(n_act) * 100.0           # Increased regularization
    W_fc = np.eye(n_contacts * 3) * 0.1
    W_delta = np.eye(n_kp * 3) * 1e5
    W_virtual = np.eye(6) * 1e3
    W_z = np.eye(n_contacts) * 1e4

    # Add small regularization to mass matrix to improve conditioning
    M_reg = M + np.eye(n_dof) * 1e-8

    cost = (
        cp.quad_form(ddq, W_ddq) +
        cp.quad_form(tau, W_tau) +
        cp.quad_form(fc, W_fc) +
        cp.quad_form(delta, W_delta) +
        cp.quad_form(tau_virtual, W_virtual)
        # + cp.quad_form(delta_z, W_z)
    )

    # --- 10. RISOLUZIONE ---
    problem = cp.Problem(cp.Minimize(cost), constraints)

    try:
        # Try multiple solvers with different configurations for better robustness
        solver_options = [
            {'solver': cp.OSQP, 'warm_start': True, 'scaling': 25, 'adaptive_rho': True, 'rho': 0.01, 'polish': True, 'polish_refine_iter': 5},
            {'solver': cp.OSQP, 'warm_start': True, 'scaling': 10, 'adaptive_rho': False, 'rho': 0.1, 'polish': False},
            {'solver': cp.ECOS, 'max_iters': 1000, 'abstol': 1e-6, 'reltol': 1e-6}
        ]

        success = False
        for i, options in enumerate(solver_options):
            try:
                problem.solve(**options)
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    success = True
                    break
            except Exception as solver_error:
                print(f"Solver {i+1} failed: {solver_error}")
                continue

        if not success:
            raise Exception("All solvers failed")

    except Exception as e:
        print(f"QPID ha fallito con un errore: {e}")
        # If we get an ArpackError, try with different regularization
        if "ArpackError" in str(type(e)) or "arpack" in str(e).lower():
            print("Tentativo di risoluzione con regolarizzazione aumentata")
            # Try with more aggressive regularization
            W_ddq = np.eye(n_dof) * 10.0  # Increased regularization
            W_tau = np.eye(n_act) * 100.0  # Increased regularization

            cost = (
                cp.quad_form(ddq, W_ddq) +
                cp.quad_form(tau, W_tau) +
                cp.quad_form(fc, W_fc) +
                cp.quad_form(delta, W_delta) +
                cp.quad_form(tau_virtual, W_virtual)
            )

            problem = cp.Problem(cp.Minimize(cost), constraints)
            try:
                problem.solve(solver=cp.OSQP, warm_start=True, scaling=25, adaptive_rho=True, rho=0.01, polish=True, polish_refine_iter=5)
            except Exception as e2:
                print(f"Regolarizzazione aumentata non ha funzionato: {e2}")
        else:
            # For other errors, just print diagnostics
            pass

        # Se fallisce, stampa la diagnostica sulle matrici "tagliate" che ha usato
        diagnose_matrix("J_c_sliced", J_c)
        diagnose_matrix("M_sliced", M)
        return None

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Attenzione: l'ottimizzatore ha fallito ({problem.status})")
        # If optimization fails, try to return at least a partial solution
        if ddq.value is not None and tau.value is not None:
            # Even if not optimal, return what we have
            pass
        else:
            return None
        
    # --- 11. RECOSTRUZIONE ARRAYS FULL-SIZE ---
    # Ricostruiamo vettori di dimensione 49 (e 43) inserendo zero ai DOF disabilitati
    ddq_out = np.zeros(n_dof_full)
    if ddq.value is not None:
        ddq_out[active_dofs] = ddq.value
    else:
        print("Warning: ddq.value is None, using zeros")

    tau_out = np.zeros(n_act_full)
    if tau.value is not None:
        for i, active_dof_idx in enumerate(active_act_dofs):
            tau_out[active_dof_idx - 6] = tau.value[i]
    else:
        print("Warning: tau.value is None, using zeros")

    return {
        "ddq": ddq_out,
        "tau": tau_out,
        "fc": fc.value,
        "delta": delta.value,
        "contact_info": contact_info
    }


s = DynamicSkeleton()
s.hip_correction = False
s12.load_from_numpy(target[0,:].reshape(-1,3),s.kps)
s.load_from_BODY12(s12)
s.exact_scale()
T_pose = s._nimble.getPositions()

kps = list(markers_dict.keys())
        
kf = None

# For stats
Q = []
Q_dot = []
Q_dotdot = []
s._nimble.setGravity([0,0,-9.81])

dt = 0.033
results = []
Fc = []
MPJPE = []
for i in range(0,target.shape[0]):
    s12.load_from_numpy(target[i,:].reshape(-1,3),s.kps)
    target_kps = s12.to_numpy(s.kps,3).reshape(1,-1).squeeze()

    result = qpid(s, target_kps,dt) 

    if result is None or result["ddq"] is None:
        print(f"[Step {i}] QP fallito, tentativo di fallback con soluzione zero")
        # Fallback: use zero accelerations and forces
        ddq_opt = np.zeros(n_dof_full)
        tau_opt = np.zeros(n_act_full)
        fc_opt = np.zeros(n_contacts * 3)

        # Create a minimal result structure for fallback
        result = {
            "ddq": ddq_opt,
            "tau": tau_opt,
            "fc": fc_opt,
            "delta": np.zeros(n_kp * 3),
            "contact_info": contact_info
        }
        results.append(result)
        continue


    results.append(result)

    ddq_opt = result["ddq"]
    
    q_current = s._nimble.getPositions()
    dq_current = s._nimble.getVelocities()
    dq_next = dq_current * 0.99 + ddq_opt * dt  # smorzamento 1%
    q_next = q_current * 0.99 + dq_next * dt  # smorzamento 1%
    
    # For debug info
    contact_estimated = np.array(get_world_contact_points(estimate_contact_points(s,target_kps)))

    contact_info_estimated = estimate_contact_points(s,target_kps)
    
    Q.append(q_current)
    Q_dot.append(dq_current)
    Q_dotdot.append(s._nimble.getAccelerations())
    s._nimble.setPositions(q_next)
    s._nimble.setVelocities(dq_next)
    s._nimble.setAccelerations(ddq_opt)
    s._nimble.setControlForces(np.hstack([np.zeros(6),result["tau"]]))

    contact = np.array(get_world_contact_points(result["contact_info"]))

    pos3D = s.to_numpy()
    error = np.linalg.norm(pos3D.reshape(-1,3)-target[i,:].reshape(-1,3), axis=1)
    MPJPE.append(np.round(np.mean(error),2))
    avg_MPJPE = np.mean(MPJPE)
print(avg_MPJPE)
