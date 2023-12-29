from Skeleton import Skeleton,ConstrainedSkeleton
import numpy as np
import pandas as pd


# df = pd.read_csv('test.csv')
df = pd.read_csv('tmp/openpose-valid.csv',sep=";")
# df = pd.read_csv('tmp/test.csv',sep=";")

s12 = Skeleton('BODY12.xml')

labels      = ['LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
labels_out  = ['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']

columns = []
for label in labels:
    columns.append(label+":X")
    columns.append(label+":Y")
    columns.append(label+":Z")
    columns.append(label+":U")
    columns.append(label+":V")

table = []
for t in range(df.shape[0]):
    
    x = df[columns].loc[t,:].to_numpy().reshape(-1,5)

    if not np.any(np.isnan(x)):
        to_output = True
        s12.load_from_numpy(x,labels)

        s15 = ConstrainedSkeleton('BODY15_constrained.xml')

        s15.load_from_BODY12(s12)

        s15.relative_position()
        
        # print(s15.estimate_height())
        
        s15.constrain()
        
        print(s15.estimate_height())
        
        # print([str(b) for b in s15.bones_list])
        # print(s15)
        
        s15.absolute_position()

        row = s15.to_numpy(labels).ravel().tolist()

    else:
        to_output = False
        row = x.ravel().tolist()
        
    columns_out = []
    for label in labels_out:
        columns_out.append(label+"_x")
        columns_out.append(label+"_y")
        columns_out.append(label+"_z")
        columns_out.append(label+"_u")
        columns_out.append(label+"_v")

    columns_out=["idx","timestamp","frame_id","body_id"]+columns_out
    if to_output:
        table.append(df.iloc[t,0:4].values.tolist() + row)

df_out = pd.DataFrame(data=np.array(table),columns = columns_out)

df_out.to_csv("openpose-valid_constrained.csv",sep=",",na_rep="nan")

#print(df_out)
# print([ str(s.length) for s in s15.bones_list])


#print(s15)
#s15.absolute_position()
#print(s15.keypoints_list[-2],s15.keypoints_list[-1],s15.keypoints_list[-1].pos-s15.keypoints_list[-2].pos)

#print([str(b) for b in s15.bones_list])



# print([ str(s.length) for s in s15.bones_list])


# print([ str(s.length) for s in s15.bones_list])


#print(s15.to_numpy(["RHip"]))

