import torch
import numpy as np
import argparse
# from tqdm.autonotebook import tqdm
import os
import torch
from cnn import Full as net
import time
import torch
import torchvision
import tensorrt as trt
from copy import deepcopy
from collections import OrderedDict, namedtuple
import torch.nn as nn
import cv2


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
class TRT(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open('pretrained/model_best.engine', 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            print(name)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
    def forward(self, im):
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]


trt=TRT()
# img=cv2.imread('img_.jpg')
# img = cv2.resize(img, (640, 360))
# img = img[:, :, ::-1].transpose(2, 0, 1)
# img = np.ascontiguousarray(img)
# img=torch.from_numpy(img)
# img = torch.unsqueeze(img, 0)  # add a batch dimension
# img=img.cuda().float() / 255.0
img = torch.rand((16,3,360,640))
img = img.cuda()
for i in range(100):
    t = time_sync()
    out=trt(img)
    print(16/(time_sync()-t))
_,da_predict=torch.max(out[0], 1)
_,ll_predict=torch.max(out[1], 1)

DA = da_predict.byte().cpu().data.numpy()[0]*255
LL = ll_predict.byte().cpu().data.numpy()[0]*255

print(LL)

cv2.imwrite('DA.jpg',DA)
cv2.imwrite('LL.jpg',LL)