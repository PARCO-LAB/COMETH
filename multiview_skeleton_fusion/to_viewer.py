from typing import List, Literal, Optional
import os 
from tap import Tap
class ArgumentParser(Tap):
    input: str="-"
    """Source JSON data (full file). If -, it reads from stdin"""
    output: str="-"
    """If - it means stdout. Otherwise, filepath to write"""
    no_rotate: bool=False
    """Avoid the 90 degree rotation on Y axis"""
    rotation: int=-90
    # pare: Optional[str]
    # """Filepath to load PARE data if existing"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, underscores_to_dashes=True, **kwargs)
    
    def process_args(self):
        # Validate arguments
        if self.input == '-':
            raise ValueError('Sorry, not supported at the moment')
    def _add_arguments(self) -> None:
        self.add_argument("input")# make input positional
        super()._add_arguments()
        

parser = ArgumentParser(
        description="Motion viewer - JSON converter", 
        epilog="Stefano Aldegheri")
args = parser.parse_args()



import pandas as pd
import numpy as np
import json
import cv2

from scipy.spatial.transform import Rotation as R


pare_people = {}

def preprocess(pare):
    global pare_people

    for jj in pare.keys():
        pare[jj]['frame_ids']
        for idx, fid in enumerate(pare[jj]['frame_ids']):
            pare_people[fid] = {
                    "smpl_joints3d": pare[jj]['joints3d'][idx],
                    "pose": pare[jj]['pose'][idx]
                }

def main():
    source_file = args.input

    # df = pd.read_csv(source_file, index_col=0, delimiter=';')
    with open(source_file) as f:
        data = json.load(f)

    basename = os.path.splitext(source_file)[0]

    # Y-up world. Rotate -90 degrees on X axis
    from_ice_to_y_up = np.eye(4)
    if not args.no_rotate:
        from_ice_to_y_up[:3,:3] = R.from_euler("xyz", [np.deg2rad(args.rotation), 0, 0], degrees=False).as_matrix()
    

    out = []
    for d in data:
        o = {
            "frame_id": d['frame_id'],
            "timestamp": (d['timestamp'] - data[0]['timestamp']) * 1e6,
            "scene": []
        }

        track_ids = d['track_ids'] if 'track_ids' in d else range(len(d['kp3d']))
        idx = 0
        for p2d, p3d, idx in zip(d['kp2d'], d['kp3d'], track_ids):
            if idx > -1:
                # idx = idx - 10
                obj = {
                    "body_id": idx,
                    "joints": {},
                    "kp2d": {},
                }
                idx = idx+1
                # print(idx,len(p2d), len(p3d))
                print(d['frame_id'],idx,p2d,p3d)
                for joint in p3d.keys():
                    point = np.array(p3d[joint][:3])
                    point = np.append(point, 1)
                    
                    point_rotated = from_ice_to_y_up @ point
                    
                    obj["joints"][joint] = {
                        "x": point_rotated[0],
                        "y": point_rotated[1],
                        "z": point_rotated[2]
                    }
                    if joint in p2d:
                        obj["joints"][joint]["u"] = p2d[joint][0]
                        obj["joints"][joint]["v"] = p2d[joint][1]
                        obj["joints"][joint]["acc"] = p2d[joint][2]
                for joint in p2d.keys():
                    obj["kp2d"][joint] = p2d[joint]
                
                # average two joints if presents, otherwise set it to none
                def avg(dst, j1, j2):
                    if j1 in obj["joints"] and j2 in obj["joints"]:
                        obj["joints"][dst] = {
                            "x": (obj["joints"][j1]["x"] + obj["joints"][j2]["x"])/2,
                            "y": (obj["joints"][j1]["y"] + obj["joints"][j2]["y"])/2,
                            "z": (obj["joints"][j1]["z"] + obj["joints"][j2]["z"])/2
                        }
                    else:
                        if j1 in obj["joints"]:
                            obj["joints"][dst] = {
                                "x": obj["joints"][j1]["x"],
                                "y": obj["joints"][j1]["y"],
                                "z": obj["joints"][j1]["z"]
                            }
                        elif j2 in obj["joints"]:
                            obj["joints"][dst] = {
                                "x": obj["joints"][j2]["x"],
                                "y": obj["joints"][j2]["y"],
                                "z": obj["joints"][j2]["z"]
                            }
                        else:
                            obj["joints"][dst] = {
                                "x": None,
                                "y": None,
                                "z": None
                            }
                # append additional points
                avg("hip", "left_hip", "right_hip")
                avg("neck", "left_shoulder", "right_shoulder")

                def augment(src, dst):
                    # to avoid doing this, is simpler to do one liner
                    # if "neck" in obj["joints"]:
                    #     obj["joints"]["sternum"] = obj["joints"]["neck"]
                    if src in obj["joints"]:
                        obj["joints"][dst] = obj["joints"][src]
                # augment points
                augment("neck", "sternum")
                augment("neck", "head")
                augment("neck", "spine")
                augment("neck", "spine1")
                augment("neck", "spine2")
                augment("left_shoulder", "left_clavicle")
                augment("right_shoulder", "right_clavicle")
                
                # TODO little patch to get it working
                obj["keypoints"] = obj["joints"]
                del obj["joints"]
                o["scene"].append(obj)
        out.append(o)
    
    if args.output == '-':
        print(json.dumps(out))
    else:
        with open(args.output, 'w') as f:
            json.dump(out, f)

if __name__ == "__main__":
    main()