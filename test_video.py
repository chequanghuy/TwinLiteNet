

import torch
import numpy as np
import argparse
from tqdm.autonotebook import tqdm
import os
import os
import torch
import pickle
from cnn import Full as net
import torch.backends.cudnn as cudnn
from crfseg import CRF
import cv2
import time
import shutil

def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
def Run(model,img,crf2):
    t1 = time_sync()
    img = cv2.resize(img, (640, 360))
    img_rs=img.copy()
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()
    print("pre time: ",time_sync()-t1)
    t1 = time_sync()
    with torch.no_grad():
        img_out = model(img)
    print("infer time: ",1.0/(time_sync()-t1)," FPS")
    t1 = time_sync()
    _,da_predict=torch.max(crf2(img_out[0]), 1)
    _,ll_predict=torch.max(crf2(img_out[1]), 1)

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    LL = ll_predict.byte().cpu().data.numpy()[0]*255
    img_rs[DA>100]=[255,0,0]
    img_rs[LL>100]=[0,255,0]
    print("post time: ",time_sync()-t1)       
    return img_rs

model = net.ESPNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('pretrained/model.pth'))
model.eval()

example = torch.rand(1, 3, 360, 640)
model = torch.jit.trace(model, example)

crf2=CRF(2)
cap = cv2.VideoCapture('video_test2.mp4')
if (cap.isOpened() == False): 
    print("Unable to read camera feed")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (640,360))
while(True):
    ret, frame = cap.read()
    if ret == True: 
        frame=Run(model,frame,crf2)
        out.write(frame)
    else:
        break 
cap.release()
out.release()
cv2.destroyAllWindows()
