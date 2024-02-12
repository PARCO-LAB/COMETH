from Skeleton.Skeleton import Skeleton, KinematicSkeleton
import numpy as np
import pandas as pd
from Completor import Completor
import json

with open("tmp/patologici_4_aperti_sx.json", "r") as read_file:
    data = json.load(read_file)

#print(data["data"][0][0]["people"][0].keys())

completor = Completor('tmp/MLP_bio_h36m_cameraview_3D_absolute_onehot',3)

s12 = Skeleton('BODY12.xml')

labels      = ['LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
labels_out  = ['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']

table = []
for t in range(len(data["data"])):
    
    d = data["data"][t][0]["people"]
    x = np.zeros((len(labels),5))
    for i,kp in enumerate(labels):
        for j in range(len(d)):
            if d[j]["name"] == kp:
                if d[j]["acc"] == 0 or d[j]["z"] == 0:
                    x[i,:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
                else:
                    x[i,:] = np.array([d[j]["x"],d[j]["y"],d[j]["z"],d[j]["u"],d[j]["v"]])

    # Complete the skeleton if something is missing
    if np.any(np.isnan(x)):
        x[:,:3] = completor(x[:,:3].ravel()).reshape(-1,3)
        
    to_output = True
    
    s12.load_from_numpy(x,labels)

    s15 = KinematicSkeleton('BODY15_constrained_5D.xml')

    s15.load_from_BODY12(s12)

    s15.IK()

    # s15.relative_position()
    # s15.constrain()   
    # s15.absolute_position()
    
    exit()