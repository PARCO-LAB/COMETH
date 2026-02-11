import numpy as np
import os
import json
import pandas as pd
from utils.evaluation import *
import time
from datetime import datetime
from multiprocessing import Process, Manager

# data_path="/home/emartini/nas/MAEVE/dataset/panoptic-toolbox/trtpose3D/"

# result_path="/home/emartini/nas/MAEVE/dataset/panoptic-toolbox/results/"
result_path="tmp/"
date = datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%m')
CONTINUOUS_STATE_PARTS = [
            "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", 
            "right_knee", "left_ankle", "right_ankle", "neck"]

# Load the json for comparison
mapping = [12, 7, 10, 4,  5, 9, 6, 8, 11, 3, 14, 13]
header = ["frame_id"]+[CONTINUOUS_STATE_PARTS[m] for m in mapping]

people = {  "170915_office1": 1,
            "161029_tools1": 2,
            "161029_build1": 2,
            "160422_ultimatum1": 7,
            "170407_haggling_a1": 3,
            "161029_sports1": 2, # Not working
            "160906_band4": 3 # Not working
        }

# HOTA parameters
# step_min = 0.20
# step_max = 0.50
# step_size = 0.05
step_min = 0.05
step_max = 0.95
step_size = 0.05



# Compared methods
manager = Manager()

# All the combinantions
cameras_list = [[6], [7], [8], [9], [10]] +\
                generate_combinations([6,7,8,9,10],2)+\
                generate_combinations([6,7,8,9,10],3)+\
                generate_combinations([6,7,8,9,10],4)+\
                generate_combinations([6,7,8,9,10],5)

# cameras_list = generate_combinations([6,7,8,9,10],5)

methods = ["cometh","openptrack","befine"] # "openptrack","befine","cometh" 
sequences = ["170915_office1","161029_tools1","161029_build1","170407_haggling_a1","160422_ultimatum1"] #"170915_office1","161029_tools1","161029_build1","160422_ultimatum1","170407_haggling_a1",] # "161029_sports1"
version = "3"
for sequence_name in sequences:
    debug_list = []
    result = []
    data_path=os.path.join("tmp",sequence_name)
    print(sequence_name)
    for cameras in cameras_list:
        print("cams:",cameras)
        
        # Skip the tools cameras 8,9,10:
        if  sequence_name == "161029_tools1" and any(e in [7,8,9,10] for e in cameras):
            continue
        
        for method in methods:
            print(method)
            preprocess_time = time.time()
            # Load ground truth
            GT = {}
            with open(os.path.join(data_path,sequence_name+".gt.json"), "r") as f:
                ground_truth = json.load(f)
            for frame in ground_truth:
                GT[frame["timestamp"]] = frame        

            # Load dut file
            DUT = {}
            with open(os.path.join(data_path,sequence_name+"."+ method+"." +".".join(map(str, cameras)) +".json" ), "r") as f:
                file = json.load(f)
            for frame in file:
                DUT[frame["timestamp"]] = frame    

            ## Build the triple nested list (shape: n_frames, n_people, n_joints, 3) and IDs (shape: n_frames, n_people)
            # union of both gt and dut            
            ids = list(GT.keys())
            for frame_dut in list(DUT.keys()):
                if frame_dut not in ids:
                    ids.append(frame_dut) 
                    
            predicted_keypoints = []
            predicted_ids = []
            ground_truth_keypoints = []
            ground_truth_ids = []
            for id in sorted(ids):
                predicted_keypoints_per_frame = []
                predicted_ids_per_frame = []
                ground_truth_keypoints_per_frame = []
                ground_truth_ids_per_frame = []
                if id in GT.keys():
                    for pp in GT[id]['continuousState']:
                        s = np.array([ [np.nan,np.nan,np.nan] if not f else f for f in pp])
                        s = s[mapping,:]
                        ground_truth_keypoints_per_frame.append(s)
                    for i in GT[id]['track_ids']:
                        ground_truth_ids_per_frame.append(i)
                if id in DUT.keys():
                    for i,pp in enumerate(DUT[id]['continuousState']):
                        s = np.array([ [np.nan,np.nan,np.nan] if not f else f for f in pp])
                        s = s[mapping,:]
                        if not np.isnan(s).all():
                            predicted_keypoints_per_frame.append(s)
                    # This doesn't work for cameras
                    for I in DUT[id]['track_ids']:
                        predicted_ids_per_frame.append(I)
                
                predicted_keypoints.append(predicted_keypoints_per_frame)
                predicted_ids.append(predicted_ids_per_frame)
                ground_truth_keypoints.append(ground_truth_keypoints_per_frame)
                ground_truth_ids.append(ground_truth_ids_per_frame)

            preprocess_time = time.time() - preprocess_time
            print("pre-process time:",round(preprocess_time,2),"s")
            # Compute HOTA
            process_time = time.time()
            res = manager.list()
            procs = []
            step = step_min
            thread_id = 0
            while step <= step_max:
                proc = Process(target=hota_par, args=(res,thread_id,predicted_keypoints, predicted_ids, ground_truth_keypoints, ground_truth_ids,step))
                procs.append(proc)
                proc.start()
                step += step_size
                thread_id += 1

            for proc in procs:
                proc.join()
            
            process_time = time.time() - process_time
            print("process time:",round(process_time,2),"s")
            
            res = list(res)
            # row = [sequence_name,len(cameras),method] + list(np.nanmean(np.array(res),0)[1:])
            
            debug_list += [[sequence_name,people[sequence_name],len(cameras),",".join(map(str, cameras)),method] + r for r in res]
            
            row = [sequence_name,people[sequence_name],len(cameras),",".join(map(str, cameras)),method] + np.round(list(100*np.nanmean(np.array([r[:-1] for r in res]),0)),1).tolist()
            result.append(row)
        
            
        debug_stats = ["Sequence", "#People","#Cams","Cams", "Aggregator",  "LocA", "DetA", "DetPR", "DetRE", "AssA","AssPR","AssRE", "HOTA","step"]
        header_stats = ["Sequence", "#People","#Cams","Cams", "Aggregator",  "LocA", "DetA", "DetPR", "DetRE", "AssA","AssPR","AssRE", "HOTA"]
        df_debug = pd.DataFrame(debug_list,columns=debug_stats)
        df = pd.DataFrame(result,columns=header_stats)
        print(df[["Cams", "Aggregator", "LocA", "DetA", "AssA","HOTA"]].tail(len(methods)).to_string())
    df.to_csv(os.path.join(result_path,date+"_"+sequence_name+'_V'+version+'.csv'), index=False)  
    df_debug.to_csv(os.path.join(result_path,date+"_"+sequence_name+'_V'+version+'_debug.csv'), index=False)