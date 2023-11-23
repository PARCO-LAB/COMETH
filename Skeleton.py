import numpy as np
import xml.etree.ElementTree as ET

from Bone import Bone
from Keypoint import Keypoint
from Joint import Joint

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

    def __str__(self):
        outstr = "Skeleton (format: {s}, dim: {d})\n".format(s=self.format,d=self.dimension)   
        outstr += to_str(self.chain,0)
        return outstr

    # #kps x dim
    def load_from_numpy(self,matrix,keypoints):
        pass
    
    # from pandas Series
    def load_from_pandas(self,df):
        pass #print(df)