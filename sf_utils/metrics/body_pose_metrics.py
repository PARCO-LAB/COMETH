import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class MetricManager:
    """ 
    Class to work with the metrics
    """
    def __init__(self, s, calc: list = [], gt = None):
        self.calculated = calc
        self.gt = gt
        self.s = s

    def store_gt_ang(self, gt_ang):
        self.gt_ang = gt_ang

    def append_new_pos(self, pos):
        self.calculated.append(pos.reshape(-1,1))

    def mae(self):
        """ 
        Mean angular error over interest joints. both input shape (n_frames, n_joint, 1)
        """
        #lumbar + thorax + scapula + shoulder + elbow + wrist
        joint_idx = list(range(20,26)) + list(range(29,37)) + list(range(39,47))
        
        n_frames = len(self.calculated)
        assert(n_frames <= len(self.gt))

    
        errors = np.abs((np.array(self.calculated)[:,joint_idx].reshape(n_frames,-1)-np.array(self.gt[:n_frames])[:,joint_idx].reshape(n_frames,-1)))
        
        #return np.mean(errors[:,[joint_idx]], axis=1)
        return errors
    
    def mpjpe(self):
        """ 
        Mean per joint positional error (over only interest joints)
        """
        #wrist + glenohumeral + elbow
        #wrist_l + glenohumeral_r + elbow_l +elbow_r + wrist_r + GlenoHumeral_l'
        joint_idx = [1,3,4,6,7,9]

        pos = self.s._nimble.getPositions()

        calc_pos = []
        gt_pos = []
        n_frames = len(self.calculated)
        for idx in range(n_frames):
            self.s._nimble.setPositions(self.calculated[idx])
            calc_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))
            self.s._nimble.setPositions(self.gt[idx])
            gt_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))

        self.s._nimble.setPositions(pos)
        
        calc_pos = np.array(calc_pos).reshape(n_frames,-1,3)[:,joint_idx].reshape(n_frames,-1,3)
        gt_pos = np.array(gt_pos).reshape(n_frames,-1,3)[:,joint_idx].reshape(n_frames,-1,3)
        pos_errors = np.linalg.norm(calc_pos-gt_pos, axis = -1)
        
        
        return pos_errors 

    def mpjae(self):
        """
        Mean per joint accelerational error (over only inteset joints)
        """
        #wrist + glenohumeral + elbow
        #wrist_l + glenohumeral_r + elbow_l +elbow_r + wrist_r + GlenoHumeral_l'
        joint_idx = [1,3,4,6,7,9]
        pos = self.s._nimble.getPositions()

        calc_pos = []
        gt_pos = []
        n_frames = len(self.calculated)
        for idx in range(n_frames):
            self.s._nimble.setPositions(self.calculated[idx])
            calc_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))
            self.s._nimble.setPositions(self.gt[idx])
            gt_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))

        self.s._nimble.setPositions(pos)

        calc_pos = np.array(calc_pos).reshape(n_frames,-1,3)[:,joint_idx].reshape(n_frames,-1,3)
        gt_pos = np.array(gt_pos).reshape(n_frames,-1,3)[:,joint_idx].reshape(n_frames,-1,3)

        accel_gt = gt_pos[:-2] - 2 * gt_pos[1:-1] + gt_pos[2:]
        accel_pred = calc_pos[:-2] - 2 * calc_pos[1:-1] + calc_pos[2:]

        acc = np.linalg.norm(accel_pred - accel_gt, axis=-1)

        return acc
    
    def p_mpjpe(self):
        #wrist + glenohumeral + elbow
        #wrist_l + glenohumeral_r + elbow_l +elbow_r + wrist_l
        joint_idx = [1,3,4,6,7,9]
        pos = self.s._nimble.getPositions()

        calc_pos = []
        gt_pos = []
        n_frames = len(self.calculated)
        for idx in range(n_frames):
            self.s._nimble.setPositions(self.calculated[idx])
            calc_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))
            self.s._nimble.setPositions(self.gt[idx])
            gt_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))

        self.s._nimble.setPositions(pos)

        calc_pos = np.array(calc_pos).reshape(n_frames,-1,3)[:,joint_idx].reshape(n_frames,-1,3)
        gt_pos = np.array(gt_pos).reshape(n_frames,-1,3)[:,joint_idx].reshape(n_frames,-1,3)



        # Select masked joints: (T, M, 3)
        """ X = pred[:, mask]
        Y = gt[:,   mask]
        M = len(joint_idx) """

        # ---------- Center ----------
        muX = calc_pos.mean(axis=1, keepdims=True)  # (T,1,3)
        muY = gt_pos.mean(axis=1, keepdims=True)  # (T,1,3)

        Xc = calc_pos - muX
        Yc = gt_pos - muY

        # ---------- Covariance ----------
        # cov[t] = Xc[t]^T @ Yc[t]  → (3×M) @ (M×3)
        cov = Xc.transpose(0,2,1) @ Yc  # (T,3,3)

        # ---------- Batched SVD ----------
        U, S, Vt = np.linalg.svd(cov)

        # ---------- Rotation ----------
        R = Vt.transpose(0,2,1) @ U.transpose(0,2,1)  # (T,3,3)

        # Ensure right-handed rotation (det = +1)
        det = np.linalg.det(R)
        bad = det < 0
        if np.any(bad):
            Vt[bad, -1, :] *= -1
            R = Vt.transpose(0,2,1) @ U.transpose(0,2,1)

        # ---------- Scale (similarity transform) ----------
        varX = (Xc ** 2).sum(axis=(1,2))  # (T,)
        s = S.sum(axis=1) / (varX + 1e-8)  # (T,)

        # ---------- Translation ----------
        # t[t] = muY[t] - s[t] * (muX[t] @ R[t])
        t = muY - (muX @ R) * s[:, None, None]  # (T,1,3)

        # ---------- Apply transform to ALL joints ----------
        aligned = s[:, None, None] * (calc_pos @ R) + t  # (T,K,3)

        errors = np.linalg.norm(aligned-gt_pos, axis = -1)

        return errors

    def save_metrics(self, metrics_list: list = ['mae', 'mpjpe', 'mpjae', 'p_mpjpe'], dir: str = './metrics'):

        if not os.path.exists(dir):
            os.mkdir(dir)
        
        for metric in metrics_list:
            #match statement only from python 3.10 (quindi tocca fare sta porcata)
            if metric == 'mae':
                errors = self.mae()
            elif metric == 'mpjpe':
                errors = self.mpjpe()
            elif metric == 'mpjae':
                errors = self.mpjae()
            elif metric == 'p_mpjpe':
                errors = self.p_mpjpe()

            df = pd.DataFrame(errors)
            df.to_csv(os.path.join(dir, metric))

        df = pd.DataFrame(np.array(self.gt).reshape(-1,49))
        df.to_csv(os.path.join(dir, 'gt'))
        df = pd.DataFrame(np.array(self.calculated).reshape(-1,49))
        df.to_csv(os.path.join(dir, 'calc'))

        pos = self.s._nimble.getPositions()

        calc_pos = []
        gt_pos = []
        n_frames = len(self.calculated)
        for idx in range(n_frames):
            self.s._nimble.setPositions(self.calculated[idx])
            calc_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))
            self.s._nimble.setPositions(self.gt[idx])
            gt_pos.append(self.s._nimble.getJointWorldPositions(self.s.joints))

        self.s._nimble.setPositions(pos)

        df = pd.DataFrame(np.array(gt_pos).reshape(-1,36)) #reshape(-1,12)
        df.to_csv(os.path.join(dir, 'gt_pos'))
        df = pd.DataFrame(np.array(calc_pos).reshape(-1,36))
        df.to_csv(os.path.join(dir, 'calc_pos'))
        
        


#TODO: testing and debugging
def plot_metric(path: str, metrics: list = ['mae', 'mpjpe'], save_path:str = './metrics_prova.svg'):

    metrics_df = pd.DataFrame()
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, n*3), sharex=True)
    axes[0].set_title("Comp. filter + gt elbows and wrists")
    for i, metric in enumerate(metrics):
        metric_path = path + metric
        if metric == 'mae':
            m_mean = pd.read_csv(metric_path, index_col=0).to_numpy().reshape(-1)
        else:
            m_df = pd.read_csv(metric_path, index_col=0)
            m_mean = np.mean(m_df.to_numpy(), axis =1).reshape(-1)

        scale_factors = {
            'mae':1,
            'mpjpe': 1000,
            'p_mpjpe': 1000,
            'mpjae': 1000
        }

        labels_dict = {
            'mae': "MAE [rad]",
            'mpjpe' : "MPJPE [mm]",
            'p_mpjpe' : "Procrustes MPJPE [mm]",
            'mpjae' : "MPJAE [mm/s^2]",
        }

        metrics_df[metric] = m_mean*scale_factors[metric]


        highlight_idx = np.arange(0, len(metrics_df), 10)

        axes[i].plot(metrics_df.index, metrics_df[metric], label=labels_dict[metric], color='blue')
        axes[i].scatter(metrics_df.index[highlight_idx], metrics_df[metric].iloc[highlight_idx],
                        color='red', label='gt given', zorder=1, s=15)
        axes[i].set_ylabel(labels_dict[metric])
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)

#TODO: testing and debugging
def plot_per_joint_metric(path:str, metric:str, save_path: str = './metrics_prova.svg'):
    joint_dict = {
        0: ('left_wrist', 'blue'),
        1: ('right_wrist', 'blue'),
        2: ('left_elbow', 'orange'),
        3: ('right_elbow', 'orange'),
        4: ('left_shoulder', 'green'),
        5: ('right_shoulder', 'green')
    }

    metric_path = path + metric
 
    m_data = pd.read_csv(metric_path, index_col=0).to_numpy.reshape(-1,6)*1000
    metrics_df = pd.DataFrame()
    fig, axes = plt.subplots(1, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        metrics_df[joint_dict[i][0]]=m_data[:,i].reshape(-1)

        highlight_idx = np.arange(0, len(metrics_df), 10)

        axes[0].plot(metrics_df.index, metrics_df[joint_dict[i][0]], label=joint_dict[i][0], color=joint_dict[i][1])
        axes[0].scatter(metrics_df.index[highlight_idx], metrics_df[joint_dict[i][0]].iloc[highlight_idx],
                        color='red', label='gt given', zorder=1, s=15)
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title(metric + '[mm]')
    
    plt.tight_layout()
    plt.savefig(save_path)