from Skeleton.Skeleton import Skeleton,ConstrainedSkeleton
import numpy as np
import pandas as pd


df = pd.read_csv('openpose_valid.csv')

s12 = Skeleton('BODY12.xml')
x = df.loc[0,:].to_numpy()[1:].reshape(-1,2)

labels = ['LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
s12.load_from_numpy(x,labels)

s15 = ConstrainedSkeleton('BODY15.xml')

s15.load_from_BODY12(s12)

print(s15.to_numpy(["RHip"]))
