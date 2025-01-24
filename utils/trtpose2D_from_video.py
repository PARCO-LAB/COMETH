from numpy.linalg import inv
import csv
import os
import numpy as np
import time
import json
import cv2
import cv2.cuda
import PIL.Image
import os.path
import sys
import os
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import torchvision.transforms as transforms
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

class Model:
    def __init__(self,dnn = 'resnet'):
        self.dnn = dnn
        self.cv2_cuda = True
        #self.parts = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]
        self.parts = ["Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "Neck"]
        torch.Tensor.ndim = property(lambda x: len(x.size())) # Avoid ndim error
        #torch.backends.cudnn.enabled = False # ???

    def load_model(self):
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)
        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        if self.dnn == 'resnet':
            MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
            OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
            model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
            WIDTH = 224
            HEIGHT = 224
        else:
            MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
            OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
            model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
            WIDTH = 256
            HEIGHT = 256
        data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
        if os.path.exists(OPTIMIZED_MODEL) == False:
            model.load_state_dict(torch.load(MODEL_WEIGHTS))
            self.model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
            torch.save(self.model_trt.state_dict(), OPTIMIZED_MODEL)
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
        torch.cuda.current_stream().synchronize()
        for i in range(50):
            y = self.model_trt(data)
        torch.cuda.current_stream().synchronize()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        
        #torch.cuda.set_device(0) # ???
        
        self.device = torch.device('cuda:0')
        self.parse_objects = ParseObjects(topology)
        self.draw_objects = DrawObjects(topology)
        
        try:
            import cv2.cuda
        except:
            cv2_cuda = False


    def get_keypoint(self,humans, hnum, peaks, max_dim):
        
        kp_dict = {}
        kpoint = []
        human = humans[0][hnum]
        C = human.shape[0]
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
                peak = (j, float(peak[0]), float(peak[1]))
                kpoint.append(peak)

                # adjust width and height
                x = peak[2] * max_dim            # IF NOT CROPPED
                y = peak[1] * max_dim            # IF NOT CROPPED
                #print(peak)
                kp_dict[self.parts[j]] = [int(x),int(y)]
            else:
                peak = (j, None, None)
                kpoint.append(peak)

        return kp_dict


    def preprocess(self,image):
        torch.cuda.set_device(0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]


    def execute(self,img, width, height):
        people = []
        data = self.preprocess(img)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        for i in range(counts[0]):
            people.append(self.get_keypoint(objects, i, peaks, max(width,height)))
        return people


    def resize(self,img,w,h):
        
        height, width, channels = img.shape
        if width-height > 0:
            img = cv2.copyMakeBorder( img, 0, int(width-height), 0, 0, cv2.BORDER_CONSTANT)
        else:
            img = cv2.copyMakeBorder( img, 0, 0, 0, int(height-width), cv2.BORDER_CONSTANT)
        #print(img.shape)
        #lumGPU0 = cv2.cuda_GpuMat()
        #lumGPU0.upload(img)
        #lumGPU0 = cv2.cuda.resize(lumGPU0,(w,h),interpolation=cv2.INTER_AREA)
        #img_out = lumGPU0.download()
        img_out = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
        return img_out

m = Model()

def get_num(element):
    return len(element)

def main():
    global directory, orientation, data_to_save
    if len(sys.argv) != 3:
        print("Usage: python3.7 <video> <out_csv_path>")
        return
    video_path = sys.argv[1]
    csv_path = sys.argv[2]

    start = time.time()
    m.load_model()
    stop = time.time()
    print("CNN initial setup time:",round(stop-start,2) ,"s")


    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")   
        exit()
    
    f = open(csv_path, 'w')
    writer = csv.writer(f)

    # Write header
    header = []
    for p in m.parts:
        header.append(p+":U")
        header.append(p+":V")
        header.append(p+":C")
    writer.writerow(header)
    frames = 0
    while cap.isOpened():  
        ret, frame = cap.read()
        if ret:
            row = []
            
            height, width = frame.shape[:2]
            img = m.resize(frame,224,224)
            res = m.execute(img,width,height)
            frame+=1
            if res:
                # Order array based on number of keypoints detected
                res.sort(key=get_num,reverse=True)
                res = res[0] # Keep only the best one
                for p in m.parts:
                    if p in res:
                        row.append(res[p][0])
                        row.append(res[p][1])
                        row.append(100)
                    else:
                        row.append(np.nan)
                        row.append(np.nan)
                        row.append(0)
                writer.writerow(row)
        else:
            break
    f.close()
    cap.release()
    
if __name__ == "__main__":
    main()