import numpy as np
import xml.etree.ElementTree as ET
from Bone import Bone
from Keypoint import Keypoint
from Joint import Joint

class Skeleton():
    
    def __init__(self,config, name = None):
        start = ET.parse(config).getroot()
        if start.tag != "skeleton":
            raise Exception("skeleton not found in xml")
        self.format = start.attrib["format"]
        self.dimension = int(start.attrib["dim"])
        for part in start:
            if part.tag == "keypoints":
                self.chain = build(part[0])
            else:
                raise Exception("Error in XML formulation: keypoints tag not found")
        self.name = name
        self.keypoints_list = get_keypoints_list(self.chain)
        self.numpy_mapping = None
        
    def __str__(self):
        outstr = "Skeleton (format: {s}, dim: {d})\n".format(s=self.format,d=self.dimension)   
        outstr += to_str(self.chain,0)
        return outstr

    # #kps x dim
    def load_from_numpy(self,matrix,labels):
        if self.numpy_mapping is None:
            indici_A = {valore: indice for indice, valore in enumerate(labels)}
            self.numpy_mapping = [indici_A.get(valore) for valore in [obj.name for obj in self.keypoints_list]]
        
        for i in range(len(self.keypoints_list)):
            #print(labels[self.numpy_mapping[i]], self.keypoints_list[i].name)
            self.keypoints_list[i].pos = matrix[self.numpy_mapping[i],:]
    
    def to_numpy(self,labels = None):
        if labels:
            indici_A = {valore: indice for indice, valore in enumerate(labels)}
            self.numpy_mapping = [indici_A.get(valore) for valore in [obj.name for obj in self.keypoints_list]]
        elif not labels and not self.numpy_mapping:
            raise Exception("You must specify which keypoints do you want")
        else:
            matrix = np.full([len(self.keypoints_list),self.dimension], np.nan)
            for i in range(len(self.keypoints_list)):
                matrix[self.numpy_mapping[i],:] = self.keypoints_list[i].pos
            return matrix
    
    # from pandas Series
    def load_from_pandas(self,df):
        print("Function not yet implemented sorry :(")
        
        
def get_keypoints_list(keypoint):
    kps = [keypoint]
    for bone in keypoint.children:
        kps += get_keypoints_list(bone.dest)
    return kps
        
# Build the skeleton recursively
def build(keypoint):
    
    # Build the first keypoint
    name = keypoint.attrib["name"]
    A = Keypoint(name = name)
    # Build the children
    next = []
    for child in keypoint:
        # Build the child recursevelt
        B = build(child) 
        
        # Link A and B with a bone
        if "bone" in child.attrib:
            bone_name = child.attrib["bone"]
            is_fixed = True if child.attrib["isfixed"] == "True" else "False"
        else:
            bone_name = None
            is_fixed = False
        next.append(Bone(A,B,bone_name,is_fixed))
    
    # Link A with the list of bones connected
    A.children = next
    return A
    
def to_str(keypoint,level):
    out = str(keypoint)
    if not keypoint.children:
        out += "\n"
    
    for bone in keypoint.children:
        out += "\n" + "\t"*level
        out += str(bone) + to_str(bone.dest,level+1)
    
    return out