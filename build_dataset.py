import nimblephysics as nimble
import numpy as np
import os
from scipy.signal import savgol_filter

def build_timestamps_from_file(osim,geometry,mot,out):
    skeleton = nimble.biomechanics.OpenSimParser.parseOsim(osim,geometry).skeleton
    motion = nimble.biomechanics.OpenSimParser.loadMot(skeleton,mot)
    np.savetxt(out, motion.timestamps, delimiter=",")

def build_dataset36DOF_from_file(osim,geometry,mot,out):
    print(out)
    skeleton = nimble.biomechanics.OpenSimParser.parseOsim(osim,geometry).skeleton
    skeleton_ref = nimble.biomechanics.OpenSimParser.parseOsim(osim,geometry).skeleton
    motion = nimble.biomechanics.OpenSimParser.loadMot(skeleton_ref,mot)
    motion_refined = motion.poses.copy()
    
    nimble_joint_names = [ 'walker_knee_r', 'wrist_l', 'hip_r', 'GlenoHumeral_r', 'elbow_l', 'hip_l', 'elbow_r', 'wrist_r',  \
                        'walker_knee_l', 'GlenoHumeral_l', 'ankle_r', 'ankle_l']

    for i in range(motion_refined.shape[1]):
        skeleton_ref.setPositions(motion.poses[:,i]) # target
        target = skeleton_ref.getJointWorldPositions([skeleton_ref.getJoint(l) for l in nimble_joint_names])
        q = skeleton.getPositions()
        older_loss = np.inf
        for _ in range(1000):
            pos = np.array(skeleton.getJointWorldPositions([skeleton.getJoint(l) for l in nimble_joint_names]))
            
            error = pos - target
            loss = np.inner(error, error)
            if np.abs(older_loss - loss) < 0.00001:
                break
            older_loss = loss
            
            
            d_loss_d__pos = 2 * (pos - target)
            d_pos_d_joint_angles = skeleton.getJointWorldPositionsJacobianWrtJointPositions([skeleton.getJoint(l) for l in nimble_joint_names])
            d_loss_d_joint_angles = d_pos_d_joint_angles.T @ d_loss_d__pos
            d_loss_d_joint_angles[29:32] = 0 # L scapula
            d_loss_d_joint_angles[39:42] = 0 # R scapula
            # d_loss_d_joint_angles[20:22] = 0 # Lumbar bending and extension
            # d_loss_d_joint_angles[23:25] = 0 # Thorax bending and extension
            # d_loss_d_joint_angles[26:29] = 0 # Head bending, extension and twist
            d_loss_d_joint_angles[21] = 0 # Lumbar extension
            d_loss_d_joint_angles[24] = 0 # Thorax extension

            q -= 0.05 * d_loss_d_joint_angles            

            # Right Hip
            q[6] = max(min(q[6], 140*np.pi/180), -40*np.pi/180)     # [-40,140]
            q[7] = max(min(q[7], 45*np.pi/180), -45*np.pi/180)     # [-40,140]
            q[8] = max(min(q[8], 45*np.pi/180), -45*np.pi/180)     # [-40,140]

            # Left Hip
            q[13] = max(min(q[13], 140*np.pi/180), -40*np.pi/180)     # [-40,140]
            q[14] = max(min(q[14], 45*np.pi/180), -45*np.pi/180)     # [-40,140]
            q[15] = max(min(q[15], 45*np.pi/180), -45*np.pi/180)     # [-40,140]

            # Right Knee
            q[9] = max(min(q[9], 140*np.pi/180), -10*np.pi/180)     # [-10,140]

            # Left Knee
            q[16] = max(min(q[16], 140*np.pi/180), -10*np.pi/180)     # [-10,140]


            # Ankles
            q[10] = max(min(q[10], 55*np.pi/180), -20*np.pi/180)     # [-10,140]
            q[17] = max(min(q[17], 55*np.pi/180), -20*np.pi/180)     # [-10,140]

            # # Lumbar amd thorax bending
            q[20] = max(min(q[20], 20*np.pi/180), -20*np.pi/180)     # [-10,140]
            q[23] = max(min(q[23], 20*np.pi/180), -20*np.pi/180)     # [-10,140]
            # # # Lumbar amd thorax extension
            # q[21] = max(min(q[21], 15*np.pi/180), -50*np.pi/180)     # [-10,140]
            # q[24] = max(min(q[24], 15*np.pi/180), -50*np.pi/180)     # [-10,140]
            
            # # Lumbar amd thorax twist
            q[22] = max(min(q[22], 5*np.pi/180), -5*np.pi/180)     # [-10,140]
            q[25] = max(min(q[25], 5*np.pi/180), -5*np.pi/180)     # [-10,140]


            # Left Shoulder
            q[42] = max(min(q[42], 150*np.pi/180), 0*np.pi/180)    # [0,150]
            q[43] = max(min(q[43], 70*np.pi/180), -90*np.pi/180)    # [-70,90]
            q[44] = max(min(q[44], 180*np.pi/180), -60*np.pi/180)    #  [-60,180]
            
            # Right Shoulder
            q[32] = max(min(q[32], 0*np.pi/180), -150*np.pi/180)    # [0,150]
            q[33] = max(min(q[33], 90*np.pi/180), -70*np.pi/180)    # [-70,90]
            q[34] = max(min(q[34], 180*np.pi/180), -60*np.pi/180)    #  [-60,180]

            # elbow range of motion"
            q[45] = max(min(q[45], 154*np.pi/180), -6*np.pi/180)    # [-6,154]
            q[35] = max(min(q[35], 154*np.pi/180), -6*np.pi/180)    # [-6,154]
                

            skeleton.setPositions(q)
        motion_refined[:,i] =  skeleton.getPositions()

    # Smooth the poses
    for i in range(motion_refined.shape[0]):
        motion_refined[i,:] = savgol_filter(motion_refined[i,:], 21, 5)

    np.savetxt(out, motion_refined.transpose(), delimiter=",")

def build_dataset49DOF_from_file(osim,geometry,mot,out):
    print(out)
    skeleton = nimble.biomechanics.OpenSimParser.parseOsim(osim,geometry).skeleton
    motion = nimble.biomechanics.OpenSimParser.loadMot(skeleton,mot)
    motion_refined = motion.poses.copy()
    
    for i in range(motion_refined.shape[1]):
        skeleton.setPositions(motion.poses[:,i]) # target
        motion_refined[:,i] =  skeleton.getPositions()

    # Smooth the poses
    for i in range(motion_refined.shape[0]):
        motion_refined[i,:] = savgol_filter(motion_refined[i,:], 21, 5)

    np.savetxt(out, motion_refined.transpose(), delimiter=",")

def main():
    counter = 0
    bioamass_path = '/home/rmhri/markerless-human-perception/src/hpe/src/Biomechanical-Model/data/bioamass_v1.0/'
    subf = ["CMU","DFAUST","MPI_Limits"] # "CMU",
    for dataset in subf:
        for d in os.listdir(bioamass_path+dataset):
            for s in os.listdir(bioamass_path+dataset+"/"+d+"/ab_fits/IK"):
                counter+=1
                build_timestamps_from_file(
                    bioamass_path+dataset+"/"+d+"/ab_fits/Models/optimized_scale_and_markers.osim",
                    "/home/rmhri/markerless-human-perception/src/hpe/src/Biomechanical-Model/data/skel_models_v1.1/Geometry/",
                    bioamass_path+dataset+"/"+d+"/ab_fits/IK/"+s,
                    "data/bioamass_timestamps/"+ dataset + "_" + d + "_" + s.replace(".mot",".csv")
                )
                # build_dataset36DOF_from_file(
                #     bioamass_path+dataset+"/"+d+"/ab_fits/Models/optimized_scale_and_markers.osim",
                #     "/home/rmhri/markerless-human-perception/src/hpe/src/Biomechanical-Model/data/skel_models_v1.1/Geometry/",
                #     bioamass_path+dataset+"/"+d+"/ab_fits/IK/"+s,
                #     "data/dataset49+/"+ dataset + "_" + d + "_" + s.replace(".mot",".csv")
                # )
                # if counter > 10:
                #     exit()

if __name__ == "__main__":
    main()
    
