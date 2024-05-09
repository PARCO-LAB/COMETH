import numpy as np
from .Keypoint import Keypoint
import math

class Bone():

    src : Keypoint
    dest : Keypoint
    name : str
    is_fixed : bool
    is_absolute : bool
    
    def __init__(self, A, B, name = None, is_fixed=None) -> None:
        self.src = A 
        self.dest = B
        self.name = name
        self.is_fixed = is_fixed
        self.is_absolute = True
        self.history = []

    def __str__(self):
        if self.name:
            out =  "-[{name}: {len}]-".format(name=self.name,len=np.round(self.length,2))
        else:
            out =  "-[Bone: {len}]-".format(name=self.name,len=np.round(self.length,2))
        return out
    
    def _get_length(self):
        length = np.nan
        if self.src.pos is not None and self.src.pos is not None:
            if self.is_absolute:
                # print(self.name,self.src.pos,self.dest.pos)
                return calculate_distance(self.src.pos,self.dest.pos)
            else:
                return calculate_distance(np.zeros(self.src.pos.shape),self.dest.pos)
        else:
            return np.nan
    
    length = property(
        fget=_get_length
    )

def calculate_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")    
    if len(point1) > 3:
        point1 = point1[:3]
        point2 = point2[:3]

    #return np.linalg.norm(point2-point1)
    
    squared_diff = [(x - y)**2 for x, y in zip(point1, point2)]
    # print(squared_diff)
    return math.sqrt(sum(squared_diff))