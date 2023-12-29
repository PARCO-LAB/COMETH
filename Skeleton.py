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
        self.name = name
        self.keypoints_list = get_keypoints_list(self.chain)
        self.bones_list = get_bones_list(self.chain)
        self.numpy_mapping = None
        self.position = np.zeros(self.dimension)
        
    def __str__(self):
        outstr = "Skeleton (format: {s}, dim: {d})\n".format(s=self.format,d=self.dimension)   
        outstr += to_str(self.chain,0)
        return outstr

    def relative_position(self):
        if np.array_equal(self.position,np.zeros(self.dimension)):
            self.position = self.chain.pos
            subtract(self.chain,np.zeros(self.dimension))
            self.chain.pos = np.zeros(self.dimension)

    def absolute_position(self):
        if not np.array_equal(self.position,np.zeros(self.dimension)):
            add(self.chain,self.position)
            self.position = np.zeros(self.dimension)


    # #kps x dim
    def load_from_numpy(self,matrix,labels):
        if self.numpy_mapping is None:
            indici_A = {valore: indice for indice, valore in enumerate(labels)}
            self.numpy_mapping = [indici_A.get(valore) for valore in [obj.name for obj in self.keypoints_list]]
        
        for i in range(len(self.keypoints_list)):
            self.keypoints_list[i].pos = matrix[self.numpy_mapping[i],:]
    
    def to_numpy(self,labels = None, dim = None):
        if dim is None:
            dim = self.dimension
        if labels:
            indici_A = {valore: indice for indice, valore in enumerate([obj.name for obj in self.keypoints_list])}
            mapping = [indici_A.get(valore) for valore in labels]
            matrix = np.full([len(labels),dim], np.nan)
            for i in range(len(labels)):
                if mapping[i] is not None:
                    matrix[i,:] = self.keypoints_list[mapping[i]].pos
            return matrix
        elif not labels and not self.numpy_mapping:
            raise Exception("You must specify which keypoints do you want")
        else:
            matrix = np.full([len(self.keypoints_list),dim], np.nan)
            for i in range(len(self.keypoints_list)):
                matrix[self.numpy_mapping[i],:] = self.keypoints_list[i].pos
            return matrix


class ConstrainedSkeleton(Skeleton):
    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.BODY12_mapping = []
        # Save constraints
        start = ET.parse(config).getroot()
        self.bones_dict = {obj.name: obj for obj in self.bones_list}
        self.proportions = {}
        self.symmetry = {}
        for part in start:
            if part.tag == "constraints":
                for e in part:
                    if e.tag == "proportions":
                        for p in e:
                            self.proportions[p.attrib["bone"]] = (float(p.attrib["mean"]),float(p.attrib["std"]))
                    elif e.tag == "symmetry":
                        for p in e:
                            self.symmetry[p.attrib["bone1"]] = p.attrib["bone2"]
                            self.symmetry[p.attrib["bone2"]] = p.attrib["bone1"]
                        
    # Pass  the position of joints from a skeleton12 to a skeleton15
    def load_from_BODY12(self,s12):
        if not self.BODY12_mapping:
            l_from = [obj.name for obj in s12.keypoints_list]
            l_to = [obj.name for obj in self.keypoints_list]
            indici_A = {valore: indice for indice, valore in enumerate(l_from)}
            self.BODY12_mapping = [indici_A.get(valore) for valore in l_to]
        # Copy the elements that are the same
        for i in range(len(self.keypoints_list)):
            if self.BODY12_mapping[i] is not None:
                self.keypoints_list[i].pos = s12.keypoints_list[self.BODY12_mapping[i]].pos
        midhip = (self.keypoints_list[9].pos+self.keypoints_list[12].pos)/2
        midshoulder = (self.keypoints_list[2].pos+self.keypoints_list[5].pos)/2
        self.keypoints_list[8].pos = midhip
        self.keypoints_list[1].pos = midshoulder
        self.keypoints_list[0].pos = (midshoulder+midhip)/2

    def constrain(self):
        height = self.estimate_height()
        constraints = { c[0] : [c[1][0]*height-c[1][1]*height, c[1][0]*height+c[1][1]*height, None] for c in self.proportions.items()}
        for bone in constraints:
            if bone in self.symmetry and self.bones_dict[self.symmetry[bone]].length > constraints[bone][0] and self.bones_dict[self.symmetry[bone]].length < constraints[bone][0]:
                constraints[bone][2] = self.bones_dict[self.symmetry[bone]].length
        if height != np.nan:
            adjust(self.chain,constraints,self.symmetry)
        
        # print(vector_angle(self.bones_dict["LForearm"].src.pos[:3],self.bones_dict["LForearm"].dest.pos[:3]))
        # print(vector_angle(self.bones_dict["USpine"].src.pos[:3],self.bones_dict["USpine"].dest.pos[:3]))
    
        # exit()
        
        
    def estimate_height(self, constrained=True):
        h = []
        for p in self.proportions:
            h.append(self.bones_dict[p].length/self.proportions[p][0])
        h = np.array(h)
        if constrained:
            outside_range_mask = np.logical_or(h < 1440, h > 2000)
            h[outside_range_mask] = np.nan
        return np.nanmean(h) if not np.all(np.isnan(h)) else np.nan

def subtract(keypoint,parent):
    for bone in keypoint.children:
        subtract(bone.dest,keypoint.pos)
        bone.is_absolute = False
    keypoint.pos -= parent
    
def add(keypoint,parent):
    keypoint.pos += parent
    for bone in keypoint.children:
        add(bone.dest,keypoint.pos)
        bone.is_absolute = True


def adjust(node,constraints,symmetry):
    # s[self.num_dimension*b_i:self.num_dimension*b_i+self.num_dimension] = b.length * (B-A) / np.linalg.norm(B-A) + A
    if type(node) == Bone:
        #print(node.src.pos, node.dest.pos, node.length)
        if node.name in constraints.keys() and (node.length < constraints[node.name][0] or node.length > constraints[node.name][1]):
            # print(node.name,"to be adjusted",node.length,constraints[node.name])
            if constraints[node.name][2] is not None:
                # print(node.name,"from",node.length,"to",constraints[node.name])
                length_adj = constraints[node.name][2]
            else:
                length_adj = constraints[node.name][0] if node.length < constraints[node.name][0] else constraints[node.name][1]
            A = node.src.pos[:3]
            B = node.dest.pos[:3]
            node.dest.pos[:3] = length_adj * (B-A) / np.linalg.norm(B-A)
            
            if node.name in symmetry:
                constraints[symmetry[node.name]][2] = length_adj
            # print("\t",node.name, "adjusted to", node.length)
        adjust(node.dest,constraints,symmetry)
    else:
        for child in node.children:
            adjust(child,constraints,symmetry)

def get_keypoints_list(keypoint):
    kps = [keypoint]
    for bone in keypoint.children:
        kps += get_keypoints_list(bone.dest)
    return kps

def get_bones_list(node):
    bone = []
    if type(node) == Bone:
        bone.append(node)
        bone += get_bones_list(node.dest)
    else:
        for child in node.children:
            bone += get_bones_list(child)
    return bone
        
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
