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
        self._pos = None
        self.confidence = 1.0  # range [0,1]
        self._history = []
        
    def __str__(self):
        if self.pos is not None:
            return "({name}: {pos})".format(name=self.name,pos=np.round(self.pos,2))
        else:
            return "({name})".format(name=self.name)
    
    def _set_pos(self,pos):
        self._pos = pos
        self.history.append(pos)

    def _get_pos(self):
        return self._pos
        
    pos = property(
        fset=_set_pos,
        fget=_get_pos
    )
    
    
    def _get_history(self):
        return self._history
    
    def _set_history(self):
        self._history = []
        
    history = property(
        fset=_set_history,
        fget=_get_history
    )
    