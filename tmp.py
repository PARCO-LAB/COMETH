import pandas as pd
from COMETH import Skeleton,DynamicSkeleton
import nimblephysics as nimble
import numpy as np

s12 = Skeleton('BODY12.xml')
# BSM
s = DynamicSkeleton(config='BODY15_constrained_3D.xml')

# rajagopal_opensim = nimble.RajagopalHumanBodyModel().skeleton
# transform = rajagopal_opensim.getBodyNode('pelvis').getWorldTransform()
# # print( transform.multiply(np.array([0,0,0])) )
# t =  transform.matrix()
# # print(t)
# p = np.array([0,0,0,1])

# print(t.dot(p.transpose()))
