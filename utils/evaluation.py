import numpy as np
from scipy.optimize import linear_sum_assignment

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


def hota_3d(predicted_keypoints, predicted_ids, ground_truth_keypoints, ground_truth_ids, distance_threshold=0.5):
    """
    Calculate Higher Order Tracking Accuracy (HOTA) for 3D human pose estimation.
    
    Parameters:
    predicted_keypoints: List of predicted 3D keypoints for each frame (shape: n_frames, n_people, n_joints, 3)
    predicted_ids: List of predicted IDs for each frame (shape: n_frames, n_people)
    ground_truth_keypoints: List of ground truth 3D keypoints for each frame (shape: n_frames, n_people, n_joints, 3)
    ground_truth_ids: List of ground truth IDs for each frame (shape: n_frames, n_people)
    distance_threshold: Distance threshold for considering a keypoint as a true positive
    
    Returns:
    float: HOTA score
    """
    assert len(predicted_keypoints) == len(ground_truth_keypoints), "Number of frames must match"
    assert len(predicted_ids) == len(ground_truth_ids), "Number of frames must match"
    
    num_frames = len(predicted_keypoints)
    total_gt = sum(len(ids) for ids in ground_truth_ids)
    total_detected = 0
    total_associated = 0

    for frame_idx in range(num_frames):
        preds = predicted_keypoints[frame_idx]
        preds_ids = predicted_ids[frame_idx]
        gts = ground_truth_keypoints[frame_idx]
        gts_ids = ground_truth_ids[frame_idx]
        
        matched_pred = set()
        matched_gt = set()
        
        # Calculate Detection Accuracy (DA)
        da_frame = 0
        for gt_idx, gt_keypoints in enumerate(gts):
            for pred_idx, pred_keypoints in enumerate(preds):
                if pred_idx in matched_pred:
                    continue
                distance = calculate_3d_distance(gt_keypoints, pred_keypoints)
                if distance <= distance_threshold:
                    da_frame += 1
                    matched_pred.add(pred_idx)
                    matched_gt.add(gt_idx)
                    break
        total_detected += da_frame
        
        # Calculate Association Accuracy (AA)
        aa_frame = 0
        for gt_idx, gt_id in enumerate(gts_ids):
            if gt_idx in matched_gt:
                pred_id = preds_ids[list(matched_pred)[list(matched_gt).index(gt_idx)]]
                if pred_id == gt_id:
                    aa_frame += 1
        total_associated += aa_frame
    
    da = total_detected / total_gt if total_gt > 0 else 0
    aa = total_associated / total_gt if total_gt > 0 else 0
    # print("total_detected:",total_detected,"(",round(da*100,1),"%)")
    # print("total_associated:",total_associated,"(",round(aa*100,1),"%)")
    hota_score = np.sqrt(da * aa)
    return hota_score

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
            cost_matrix[i, j] = calculate_3d_distance(predicted_keypoints[i], ground_truth_keypoints[j])
    
    return cost_matrix

def mota_motp(predicted_keypoints, predicted_ids, ground_truth_keypoints, ground_truth_ids, distance_threshold=0.1):
    """
    Calculate MOTA and MOTP for multi-person 3D human pose estimation.
    
    Parameters:
    predicted_keypoints: List of predicted 3D keypoints for each frame (shape: n_frames, n_people, n_joints, 3)
    predicted_ids: List of predicted IDs for each frame (shape: n_frames, n_people)
    ground_truth_keypoints: List of ground truth 3D keypoints for each frame (shape: n_frames, n_people, n_joints, 3)
    ground_truth_ids: List of ground truth IDs for each frame (shape: n_frames, n_people)
    distance_threshold: Distance threshold for considering a keypoint as a true positive
    
    Returns:
    float, float: MOTA and MOTP scores
    """
    assert len(predicted_keypoints) == len(ground_truth_keypoints), "Number of frames must match"
    assert len(predicted_ids) == len(ground_truth_ids), "Number of frames must match"
    
    num_frames = len(predicted_keypoints)
    total_gt = sum(len(ids) for ids in ground_truth_ids)
    total_detections = 0
    total_false_positives = 0
    total_misses = 0
    total_switches = 0
    total_precision = 0
    total_matches = 0

    previous_matched_gt_ids = {}

    for frame_idx in range(num_frames):
        preds = predicted_keypoints[frame_idx]
        preds_ids = predicted_ids[frame_idx]
        gts = ground_truth_keypoints[frame_idx]
        gts_ids = ground_truth_ids[frame_idx]
        
        cost_matrix = compute_cost_matrix(preds, gts)
        if np.isnan(cost_matrix).any():
            cost_matrix[np.isnan(cost_matrix)] = np.nanmax(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_pred = set()
        matched_gt = set()
        matches = 0
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= distance_threshold:
                matched_pred.add(r)
                matched_gt.add(c)
                total_precision += cost_matrix[r, c]
                matches += 1
                
                pred_id = preds_ids[r]
                gt_id = gts_ids[c]
                
                if gt_id in previous_matched_gt_ids and previous_matched_gt_ids[gt_id] != pred_id:
                    total_switches += 1
                
                previous_matched_gt_ids[gt_id] = pred_id
        
        total_detections += matches
        total_misses += len(gts) - matches
        total_false_positives += len(preds) - matches
        total_matches += matches

    mota = 1 - (total_misses + total_false_positives + total_switches) / total_gt
    motp = total_precision / total_matches if total_matches > 0 else 0

    return mota, motp
