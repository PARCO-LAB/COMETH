from Skeleton import Skeleton
import numpy as np
import pandas as pd


df = pd.read_csv('openpose_valid.csv')

s1 = Skeleton('BODY12.xml')
x = df.loc[0,:].to_numpy()[1:].reshape(-1,2)

labels = ['LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
s1.load_from_numpy(x,labels)

#print(x == s1.to_numpy())