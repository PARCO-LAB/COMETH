import numpy as np

class Keypoint():
    
    name : str
    children : list
    parent : list
    pos : np.array
    
    def __init__(self, parent = None, children = None, name = None, dim = None):
        self.name = name
        self.children = children
        self.parent = parent
        self.dim = dim
        self.pos = None
        
    def __str__(self):
        if self.pos:
            return "({name}: {pos})".format(name=self.name,pos=round(self.pos,2))
        else:
            return "({name})".format(name=self.name)