import numpy as np
from Keypoint import Keypoint

class Bone():

    src : Keypoint
    dest : Keypoint
    name : str
    is_fixed : bool
    
    def __init__(self, A, B, name = None, is_fixed=None) -> None:
        self.src = A 
        self.dest = B
        self.name = name
        self.is_fixed = is_fixed

    def __str__(self):
        if self.name:
            out =  "-[{name}: {len}]-".format(name=self.name,len=self.length)
        else:
            out =  "-[Bone: {len}]-".format(name=self.name,len=self.length)
        return out
    
    def _get_length(self):
        return np.nan
    
    length = property(
        fget=_get_length
    )
    