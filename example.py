from Skeleton import Skeleton

import pandas as pd


df = pd.read_csv('openpose_valid.csv')

s = Skeleton('BODY15.xml')
s.load_from_pandas(df.iloc[0])
print(s)