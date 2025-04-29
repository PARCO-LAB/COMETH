import numpy as np

# Maximum bone length that a incoming measuremenet must have w.r.t. the skeleton
MAX_BONE_LENGTH_DIFFERENCE = 0.5   # meters

# Maximum absolute value that a incoming measured bone must have
MAX_BONE_LENGTH = 1   # meters

# Maximum distance between a keypoints and the cluster center to be considered 
# as part of the cluster
MAX_DISTANCE_FROM_CLUSTER = 2

# qpik default precision and iterations (may sensibly change the results)
QPIK_PRECISION = 0.001
QPIK_ITERATIONS = 100

# Keypoints offset from the osim standard
RASI_OFFSET = np.array([0,0.005,0.13])
LASI_OFFSET = np.array([0,0.005,-0.13])
LPSI_OFFSET = np.array([-0.14,0.015,-0.07])
RPSI_OFFSET = np.array([-0.14,0.015,+0.07])

RCAJ_OFFSET = np.array([0.015,-0.035,-0.02])
RHGT_OFFSET = np.array([-0.05,0,0])
LCAJ_OFFSET = np.array([0.015,-0.035,0.02])
LHGT_OFFSET = np.array([-0.05,0,0])

# Rajagopal convetions
RAJAGOPAL_KPS =  ['RKnee', 'LWrist', 'RHip', 'RShoulder',  'LElbow', 'LHip', 'RElbow', 'RWrist', 'LKnee', 'LShoulder', 'RAnkle', 'LAnkle']
RAJAGOPAL_JOINT_NAMES = [ 'walker_knee_r', 'radius_hand_l', 'hip_r', 'acromial_r', 'elbow_l', 'hip_l', 'elbow_r', 'radius_hand_r',  \
            'walker_knee_l', 'acromial_l', 'ankle_r', 'ankle_l']
RAJAGOPAL_BODY_DICT = { 'pelvis' : 'LPelvis',
                        'femur_r' : 'RFemur',
                        'tibia_r' : 'RTibia',
                        'talus_r' : '',
                        'calcn_r' : '',
                        'toes_r' : '',
                        'femur_l' : 'LFemur',
                        'tibia_l' : 'LTibia',
                        'talus_l' : '',
                        'calcn_l' : '',
                        'toes_l' : '',
                        'torso' : 'LClavicle',
                        'humerus_r' : 'RHumerus',
                        'ulna_r' : 'RHumerus',
                        'radius_r' : 'RForearm',
                        'hand_r' : '',
                        'humerus_l' : 'LHumerus',
                        'ulna_l' : 'LHumerus',
                        'radius_l' : 'LForearm',
                        'hand_l' : ''}

# BSM conventions
BSM_KPS =  ['RKnee', 'LWrist', 'RHip', 'RShoulder',  'LElbow', 'LHip', 'RElbow', 'RWrist', 'LKnee', 'LShoulder', 'RAnkle', 'LAnkle']
BSM_JOINT_NAMES = [ 'walker_knee_r', 'wrist_l', 'hip_r', 'GlenoHumeral_r', 'elbow_l', 'hip_l', 'elbow_r', 'wrist_r',  \
            'walker_knee_l', 'GlenoHumeral_l', 'ankle_r', 'ankle_l']
BSM_BODY_DICT = {  'pelvis':'Core', #LPelvis
                    'femur_r':'RFemur',
                    'tibia_r':'RTibia',
                    'talus_r':'',
                    'calcn_r':'',
                    'toes_r':'',
                    'femur_l':'LFemur',
                    'tibia_l':'LTibia',
                    'talus_l':'',
                    'calcn_l':'',
                    'toes_l':'',
                    'lumbar_body':'Core',#LClavicle
                    'thorax':'Core',#LClavicle
                    'head':'',
                    'scapula_r':'Core',#LClavicle
                    'humerus_r':'RHumerus',
                    'ulna_r':'RForearm',
                    'radius_r':'RForearm',
                    'hand_r':'',
                    'scapula_l':'Core',#LClavicle
                    'humerus_l':'LHumerus',
                    'ulna_l':'LForearm',
                    'radius_l':'LForearm',
                    'hand_l':''}


# Joint limits in degrees from the biomechanics literature
Q_LOWER_BOUND = np.ones((49))*(-180)
Q_UPPER_BOUND = np.ones((49))*180

Q_LOWER_BOUND[0:6] = -np.inf
Q_LOWER_BOUND[6] = -40
Q_LOWER_BOUND[7] = -45
Q_LOWER_BOUND[8] = -45
Q_LOWER_BOUND[13] =-40
Q_LOWER_BOUND[14] =-45
Q_LOWER_BOUND[15] =-45
Q_LOWER_BOUND[9] = -10
Q_LOWER_BOUND[16] =-10
Q_LOWER_BOUND[10] =-20
Q_LOWER_BOUND[17] =-20
Q_LOWER_BOUND[20] =-20
Q_LOWER_BOUND[23] =-20
Q_LOWER_BOUND[29] =-90
Q_LOWER_BOUND[39] =-90
Q_LOWER_BOUND[31] =-10
Q_LOWER_BOUND[41] =-10
Q_LOWER_BOUND[22] =-5
Q_LOWER_BOUND[25] =-5
Q_LOWER_BOUND[42] =0
Q_LOWER_BOUND[43] =-90
Q_LOWER_BOUND[44] =-60
Q_LOWER_BOUND[32] =-150
Q_LOWER_BOUND[33] =-70
Q_LOWER_BOUND[34] =-60
Q_LOWER_BOUND[45] =-6
Q_LOWER_BOUND[35] =-6
Q_LOWER_BOUND[30] = -8
Q_LOWER_BOUND[40] = -8
Q_LOWER_BOUND[21] = -5
Q_LOWER_BOUND[24] = -5

Q_UPPER_BOUND[0:6] = np.inf
Q_UPPER_BOUND[6]  = 140
Q_UPPER_BOUND[7]  = 45
Q_UPPER_BOUND[8]  = 45
Q_UPPER_BOUND[13] = 140
Q_UPPER_BOUND[31] = 40
Q_UPPER_BOUND[41] = 40
Q_UPPER_BOUND[14] = 45
Q_UPPER_BOUND[15] = 45
Q_UPPER_BOUND[9]  = 140
Q_UPPER_BOUND[16] = 140
Q_UPPER_BOUND[10] = 55
Q_UPPER_BOUND[17] = 55
Q_UPPER_BOUND[29] =-55
Q_UPPER_BOUND[39] =-55
Q_UPPER_BOUND[20] = 20
Q_UPPER_BOUND[23] = 20
Q_UPPER_BOUND[22] = 5
Q_UPPER_BOUND[25] = 5
Q_UPPER_BOUND[42] = 150
Q_UPPER_BOUND[43] = 70
Q_UPPER_BOUND[44] = 180
Q_UPPER_BOUND[32] = 0
Q_UPPER_BOUND[33] = 90
Q_UPPER_BOUND[34] = 180
Q_UPPER_BOUND[45] = 154
Q_UPPER_BOUND[35] = 154
Q_UPPER_BOUND[30] = 2
Q_UPPER_BOUND[40] = 2
Q_UPPER_BOUND[21] = 5
Q_UPPER_BOUND[24] = 5

# Lower and upper bounds for the velocities (expressed in rad/s)
# learned from BSM dataset
QDOT_LOWER_BOUND = np.array([-1,-1,-1,-2,-2,-2,-1.58,-0.57,-0.61,-1.97,0,0,0,-1.61,-0.56,-0.55,-1.97,0,0,0,-0.49,0,-0.31,-0.37,0,-0.29,0,0,0,0,0,0,-0.80,-0.84,-1.37,-1.34,-0.080,0,0,0,0,0,-0.80,-0.84,-1.42,-1.16,-0.070,0,0])
QDOT_UPPER_BOUND = np.array([1,1,1,2,2,2,1.93,0.54,0.53,2.14,0,0,0,1.95,0.54,0.51,2.23,0,0,0,0.49,0,0.32,0.38,0,0.29,0,0,0,0,0,0,0.78,0.88,1.4,1.47,0.090,0,0,0,0,0,0.84,0.72,1.37,1.27,0.080,0,0])
