from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import glob



def process_file(filename, scaler, sample=None,_type="3D"):
    scaler = StandardScaler()
    X = []   
    df = pd.read_csv(filename)
    # Rest of your processing code for df and gt ...
    # print(np.isnan(df.to_numpy()).any(),np.isnan(gt.to_numpy()).any())
    if 'time' in df:
        df = df.drop(columns = ['time'])
    if 'Unnamed: 0' in df:
        df = df.drop(columns = ['Unnamed: 0'])
    X = []
    columns_full = list(df.columns)
    # to_drop = []
    # for i in columns_full:
    #     if i.split(':')[0] not in KEYPOINTS or i.split(":")[1] not in ["X","Y","Z"]:
    #         to_drop.append(i)
    # df = df.drop(columns = to_drop)
    
    columns_full = []
    for k in KEYPOINTS:
        columns_full.append(k+":X")    
        columns_full.append(k+":Y")    
        columns_full.append(k+":Z")    
    df = df[columns_full]
    
    
    for j in range(len(df)):                
        in_seq = df.iloc[j,:].to_frame().transpose()
        in_seq = center_on_hip_first(in_seq)
        in_seq = in_seq.drop(columns=['index'])
        # in_seq = scaler.fit_transform(in_seq.transpose())
        X.append(in_seq)
    print(len(KEYPOINTS),np.array(X).shape)
    return X

def parallel_processing(files, sample=None,split=0.8,type="3D",threads=128):
    final_X = []

    # Create a ThreadPoolExecutor with a max_workers parameter to control concurrency
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for filename in files:
            scaler = StandardScaler()
            future = executor.submit(process_file, filename, scaler, sample,_type=type)
            futures.append(future)

        for future in futures:
            X = future.result()
            if len(X) > sample:
                subset_index = np.random.choice(list(range(0, len(X))),size = sample, replace = False)
                X = [X[i] for i in subset_index]
            final_X += X
        subset_index = np.random.choice(list(range(0, len(final_X))),size = len(final_X), replace = False)
        final_X = [final_X[i] for i in subset_index]
        X = np.array(final_X)

        if split < 1:
            split_index = int(len(X) * split)  # 80% training, 20% validation
            X_train, X_val = X[:split_index], X[split_index:]
        else:
            X_train, X_val = X, None
        if scaler:
            return X_train, X_val, None, None, scaler
        else:
            return X_train, X_val, None, None, None

def center_on_hip_first(df : pd.DataFrame, kp_center = 'Hip'):
    df.reset_index(inplace = True)
    return df



KEYPOINTS = ['LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']


in_path = "/home/emartini/nas/MAEVE/HUMAN_MODEL/dataset/DDPM/gt/"
out_npy = "/home/emartini/nas/MAEVE/HUMAN_MODEL/dataset/training_FLK2/h36m_cameraview_3D"
_sample = 1000
# exist or not.
if not os.path.exists(out_npy):
    os.makedirs(out_npy)
scaler = 1
files = sorted(glob.glob(in_path + '/*.csv'))
X_train, X_val, _, _, scaler =  parallel_processing(files, _sample,split=0.8,type="3D",threads=64)
if np.any(X_train):
    np.save( out_npy + '/X_train_' + str(_sample)+  '.npy', X_train)
    np.save( out_npy + '/X_val_' + str(_sample)+  '.npy', X_val)
    