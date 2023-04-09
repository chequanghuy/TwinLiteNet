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
def Run(model,img,crf2,DAl,LLl):
    t1 = time_sync()
    img = cv2.resize(img, (640, 360))
    DAl = cv2.resize(DAl, (640, 360))
    LLl = cv2.resize(LLl, (640, 360))

    DAl=cv2.cvtColor(DAl,cv2.COLOR_BGR2GRAY)
    LLl=cv2.cvtColor(LLl,cv2.COLOR_BGR2GRAY)
    img_rs=img.copy()
    img_rs2=img.copy()
    img_rs3=img.copy()
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
    x0=img_out[0]
    x1=img_out[1]
    for i in range(10):
        x0=crf2(x0)
        x1=crf2(x1)
    _,da_predict=torch.max(x0, 1)
    _,ll_predict=torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    LL = ll_predict.byte().cpu().data.numpy()[0]*255
    # img_rs[DA>100]=[255,0,0]
    img_rs[LL>100]=[0,255,0]
    

    _,da_predict2=torch.max(img_out[0], 1)
    _,ll_predict2=torch.max(img_out[1], 1)

    DA2 = da_predict2.byte().cpu().data.numpy()[0]*255
    LL2 = ll_predict2.byte().cpu().data.numpy()[0]*255
    img_rs2[DA2>100]=[255,0,0]
    img_rs2[LL2>100]=[0,255,0]
    print("post time: ",time_sync()-t1)    
    img_rs3[DAl>100]=[255,0,0]
    img_rs3[LLl>100]=[0,255,0]
    im_v = cv2.vconcat([img_rs, img_rs2])   
    im_v = cv2.vconcat([im_v, img_rs3])   
    return img_v


model = net.ESPNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('pretrained/model.pth'))
model.eval()


crf2=CRF(2)
image_list=os.listdir('/home/ceec/huycq/data/bdd100k/images/val')
shutil.rmtree('visualize_re')
os.mkdir('visualize_re')
for i, imgName in enumerate(image_list[200:300]):
    img = cv2.imread(os.path.join('/home/ceec/huycq/data/bdd100k/images/val',imgName))
    label_name=imgName.replace("jpg","png")
    DA = cv2.imread(os.path.join('/home/ceec/huycq/data/bdd100k/segments/val',label_name))
    LL = cv2.imread(os.path.join('/home/ceec/huycq/data/bdd100k/lane/val',label_name))
    img=Run(model,img,crf2,DA,LL)
    cv2.imwrite(os.path.join('visualize_re',imgName),img)