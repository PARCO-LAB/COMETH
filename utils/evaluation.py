import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import combinations

def generate_combinations(values, n):
    return [list(comb) for comb in combinations(values, n)]

def MPJPE(s,gt):
    distances = np.linalg.norm(s - gt, axis=1)
    return distances

def calculate_3d_distance(keypoints1, keypoints2):
    """
    Calculate the average Euclidean distance between two sets of 3D keypoints.
    
    Parameters:
    keypoints1, keypoints2: Arrays of shape (n_joints, 3)
    
    Returns:
    float: Average Euclidean distance
    """
    # print("debug")
    # print(keypoints1)
    # print(keypoints2)
    return np.nanmean(np.linalg.norm(keypoints1 - keypoints2, axis=1))


# score in [0,1]
def score(A,B):
    return max(0,1-calculate_3d_distance(A,B))

def locA(set):
    pass

# return AssA, AssPr, AssRe
# assre: measures how well predicted trajectories cover ground-truth trajectories. 
#        Low AssRe will result when a tracker splits an object up into multiple predicted tracks
# asspr: how well predicted trajectories keep to tracking the same ground-truth trajectories. 
#        A fail will result if a predicted track extends over multiple objects
def assA(TP,totIDs):
    
    assre = 0
    asspr = 0
    
    for s in TP:
        TPA = 0
        FNA = 0
        FPA = 0
        
        gtID = s[0]
        prID = s[1]
        for k in totIDs:
            if gtID == k[0] and prID == k[1]:
                # print(s,"->",k,"TPA")
                TPA += 1
            elif gtID != k[0] and prID == k[1]:
                # print(s,"->",k,"FPA")
                FPA += 1
            elif gtID == k[0] and prID != k[1]:
                # print(s,"->",k,"FNA")
                FNA += 1

        assre += TPA / (FNA + TPA)
        asspr += TPA / (FPA + TPA)
        
        # print(TPA,FNA,FPA)
    
    assre/=len(TP)
    asspr/=len(TP)
    
    
    assa = (assre*asspr) / (assre + asspr - assre*asspr)
    
    return assa, asspr, assre
                
    
def hota_par(grid, index, predicted_keypoints, predicted_ids, ground_truth_keypoints, ground_truth_ids, alpha=0.5):
    grid.append(hota(predicted_keypoints, predicted_ids, ground_truth_keypoints, ground_truth_ids, alpha))
    
def hota_base_par(grid, index, predicted_keypoints, ground_truth_keypoints, alpha=0.5):
    grid.append(hota_base(predicted_keypoints, ground_truth_keypoints, alpha))
    
    
def hota(predicted_keypoints, predicted_ids, ground_truth_keypoints, ground_truth_ids, alpha=0.5):
    """
    Calculate Higher Order Tracking Accuracy (HOTA) for 3D human pose estimation.
    
    Parameters:
    predicted_keypoints: List of predicted 3D keypoints for each frame (shape: n_frames, n_people, n_joints, 3)
    predicted_ids: List of predicted IDs for each frame (shape: n_frames, n_people)
    ground_truth_keypoints: List of ground truth 3D keypoints for each frame (shape: n_frames, n_people, n_joints, 3)
    ground_truth_ids: List of ground truth IDs for each frame (shape: n_frames, n_people)
    alpha: Distance threshold for considering a keypoint as a true positive
    
    Returns:
    list of floats: HOTA score and its components
    """
    assert len(predicted_keypoints) == len(ground_truth_keypoints), "Number of frames must match"
    assert len(predicted_ids) == len(ground_truth_ids), "Number of frames must match"
    
    num_frames = len(predicted_keypoints)
    total_gt = sum(len(ids) for ids in ground_truth_ids)
    total_pr = sum(len(ids) for ids in predicted_ids)

    tp = 0
    fn = 0
    fp = 0
    matched_sk = []
    matched_id = []
    totIDs = []
    
    loca = 0
    
    for frame_idx in range(num_frames):
        prSK = predicted_keypoints[frame_idx]
        prID = predicted_ids[frame_idx]
        gtSK = ground_truth_keypoints[frame_idx]
        gtID = ground_truth_ids[frame_idx]
        id_dic = {}
        # base cases
        if prID and not gtID:
            totIDs += [(-1,id) for id in prID]
        if gtID and not prID:
            totIDs += [(id,-1) for id in gtID]
        if len(prID) > len(gtID):
            fp += len(prID)-len(gtID)
        elif len(prID) < len(gtID):
            fn += len(gtID)-len(prID)
        #lsa
        else:
            # size = max(len(prID),len(gtID))
            cost_matrix = compute_cost_matrix(prSK,gtSK)
            row_ind, col_ind = linear_sum_assignment(-cost_matrix) # Negate for minimization
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] >= alpha:
                    tp += 1
                    matched_sk.append((gtSK[c], prSK[r]))
                    matched_id.append((gtID[c], prID[r]))
                    loca += score(prSK[r],gtSK[c])
                    # id_dic[gtID[c]] = prID[r]
                # Not sure about this else below :-)
                    totIDs.append((gtID[c], prID[r]))
                else:
                    totIDs.append((gtID[c], -1))
                    totIDs.append((-1,prID[r]))
                    fp += 1
                    fn += 1
    
    loca = loca/tp if tp != 0 else np.nan
    
    detre = tp/(tp+fn)    
    detpr = tp/(tp+fp)    
    deta = tp/(tp+fp+fn)
            
    if tp != 0:
        assa,asspr,assre = assA(matched_id,totIDs)
    else:
        assa, asspr, assre = np.nan, np.nan, np.nan
    return [loca, deta, detpr, detre, assa,asspr,assre,np.sqrt(assa*deta),alpha]

def hota_base(predicted_keypoints, ground_truth_keypoints, alpha=0.5):
    
    assert len(predicted_keypoints) == len(ground_truth_keypoints), "Number of frames must match"
    
    
    num_frames = len(predicted_keypoints)
    
    tp = 0
    fn = 0
    fp = 0
    matched_sk = []
    
    loca = 0
    
    for frame_idx in range(num_frames):
        prSK = predicted_keypoints[frame_idx]
        gtSK = ground_truth_keypoints[frame_idx]
        
        # base cases
        if len(prSK) > len(gtSK):
            fp += len(prSK)-len(gtSK)
        elif len(prSK) < len(gtSK):
            fn += len(gtSK)-len(prSK)
        #lsa
        else:
            cost_matrix = compute_cost_matrix(prSK,gtSK)
            row_ind, col_ind = linear_sum_assignment(-cost_matrix) # Negate for minimization
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] >= alpha:
                    tp += 1
                    matched_sk.append((gtSK[c], prSK[r]))
                    loca += score(prSK[r],gtSK[c])
                else:
                    fp += 1
                    fn += 1
    
    loca = loca/tp if tp != 0 else np.nan
    
    detre = tp/(tp+fn)    
    detpr = tp/(tp+fp)    
    deta = tp/(tp+fp+fn)
            
    assa, asspr, assre = np.nan, np.nan, np.nan
    return [loca, deta, detpr, detre, assa,asspr,assre,np.nan,alpha]
    

def compute_cost_matrix(predicted_keypoints, ground_truth_keypoints):
    """
    Compute the cost matrix for matching predicted keypoints to ground truth keypoints.
    
    Parameters:
    predicted_keypoints: List of predicted 3D keypoints (shape: n_people, n_joints, 3)
    ground_truth_keypoints: List of ground truth 3D keypoints (shape: n_people, n_joints, 3)
    
    Returns:
    np.ndarray: Cost matrix (shape: n_pred, n_gt)
    """
    n_pred = len(predicted_keypoints)
    n_gt = len(ground_truth_keypoints)
    cost_matrix = np.zeros((n_pred, n_gt))
    
    for i in range(n_pred):
        for j in range(n_gt):
            # cost_matrix[i, j] = 1-score(predicted_keypoints[i], ground_truth_keypoints[j])
            cost_matrix[i, j] = score(predicted_keypoints[i], ground_truth_keypoints[j])
    
    
    
    return cost_matrix

