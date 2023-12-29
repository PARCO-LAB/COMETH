from Skeleton import Skeleton,ConstrainedSkeleton
import numpy as np
import pandas as pd
from Completor import Completor

# df = pd.read_csv('test.csv')
df = pd.read_csv('tmp/openpose-valid.csv',sep=";")
# df = pd.read_csv('tmp/test_missing.csv',sep=";")

df_in = df.copy()

completor = Completor('tmp/MLP_bio_h36m_cameraview_3D_absolute_onehot',3)

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

    # Complete the skeleton if something is missing
    if np.any(np.isnan(x)):
        x[:,:3] = completor(x[:,:3].ravel()).reshape(-1,3)
        
    to_output = True
    
    s12.load_from_numpy(x,labels)

    s15 = ConstrainedSkeleton('BODY15_constrained.xml')

    s15.load_from_BODY12(s12)

    #s15.relative_position()
    #s15.constrain()   
    #s15.absolute_position()

    row = s15.to_numpy(labels).ravel().tolist()


    # to_output = False
    # row = x.ravel().tolist()
        
    columns_out = []
    for label in labels_out:
        columns_out.append(label+"_x")
        columns_out.append(label+"_y")
        columns_out.append(label+"_z")
        columns_out.append(label+"_u")
        columns_out.append(label+"_v")

    columns_out = ["idx","timestamp","frame_id","body_id"] + columns_out
    if to_output:
        table.append(df.iloc[t,0:4].values.tolist() + row)


# print(df_in.columns)

# df_in = pd.DataFrame(data=df_in.values,columns = columns_out)

df_out = pd.DataFrame(data=np.array(table),columns = columns_out)
#df_out["body_id"] = 10



# tbm = [df_in,df_out]
# result = pd.concat(tbm)
result = df_out
result.to_csv("openpose-valid_filled.csv",sep=",",na_rep="nan")
