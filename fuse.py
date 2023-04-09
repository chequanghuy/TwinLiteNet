import sys
sys.path.append('/home/ceec/huycq/HybridNets')
import torch
import numpy as np
import torch.nn as nn
import argparse
from tqdm.autonotebook import tqdm
import os
from IOUEval import iouEval,SegmentationMetric
import loadData as ld
import os
import torch
import pickle
from cnn import Full as net
import torch.backends.cudnn as cudnn
import Transforms as myTransforms
import DataSet as myDataLoader
from argparse import ArgumentParser
from train_utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from utils import smp_metrics
from crfseg import CRF
from utils.constants import *
import cv2
import time
import shutil
from cnn.Full import CBR,UPx2
def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv
# def fuse_deconv_and_bn(conv, bn):
#     # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
#     # print(conv.in_channels)
#     # print(conv.in_channels,
#     #     conv.out_channels,
#     #     conv.kernel_size,
#     #     conv.stride,
#     #     conv.padding,
#     #     conv.groups)
#     fuseddeconv = nn.ConvTranspose2d(conv.in_channels,
#                           conv.out_channels,
#                           kernel_size=conv.kernel_size,
#                           stride=conv.stride,
#                           padding=conv.padding,
#                           groups=conv.groups,
#                           bias=True).requires_grad_(False).to(conv.weight.device)

#     # prepare filters
#     w_conv = conv.weight.clone().view(conv.out_channels, -1)
#     w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
#     # print("bn weight :",w_bn.size())
#     fuseddeconv.weight.copy_(torch.mm(w_bn, w_conv).view(fuseddeconv.weight.shape))

#     # prepare spatial bias
#     # print("decon",conv.weight.size())
#     b_conv = torch.zeros(conv.weight.size(1), device=conv.weight.device) if conv.bias is None else conv.bias
#     b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
#     # print(w_bn.size(), b_conv.size())
#     fuseddeconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

#     return fuseddeconv



def fuse_deconv_and_bn(conv_transpose, bn):
    fusedconv = nn.ConvTranspose2d(conv_transpose.in_channels,
                                    conv_transpose.out_channels,
                                    kernel_size=conv_transpose.kernel_size,
                                    stride=conv_transpose.stride,
                                    padding=conv_transpose.padding,
                                    groups=conv_transpose.groups,
                                    bias=True).requires_grad_(False).to(conv_transpose.weight.device)

    # prepare filters
    w_conv_transpose = conv_transpose.weight.clone().view(conv_transpose.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv_transpose.transpose(0, 1)).transpose(0, 1).view(fusedconv.weight.shape))
    b_conv_transpose = torch.zeros(conv_transpose.weight.size(1), device=conv_transpose.weight.device) if conv_transpose.bias is None else conv_transpose.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(b_conv_transpose.reshape(-1, 1), w_bn).reshape(-1) + b_bn)

    return fusedconv




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = net.ESPNet()
        self.net = torch.nn.DataParallel(self.net)
        self.net = self.net.cuda()
        self.net.load_state_dict(torch.load('/home/ceec/huycq/ESPNetv2/segmentation/cus_1/model_70.pth'))
        self.net.eval()
        print("load done")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.net.modules():
            if type(m) is CBR and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif type(m) is UPx2 and hasattr(m, 'bn'):
                m.deconv = fuse_deconv_and_bn(m.deconv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        # info(model)
        return self
    
    
    def forward(self,x):
        return self.net(x)
@torch.no_grad()
def val(val_loader, model):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    #switch to evaluation mode
    model.eval()
    crf2=CRF(2)
    # iouEvalVal = iouEval(args.classes)

    DA=SegmentationMetric(2)
    LL=SegmentationMetric(2)

    epoch_loss = []
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()
    total_batches = len(val_loader)
    
    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.cuda().float() / 255.0
            # target = target.cuda()

        input_var = input
        target_var = target

        # run the mdoel
        with torch.no_grad():
            output = model(input_var)

        out_da,out_ll=output
        target_da,target_ll=target

        _,da_predict=torch.max(out_da, 1)
        _,da_gt=torch.max(target_da, 1)

        _,ll_predict=torch.max(out_ll, 1)
        _,ll_gt=torch.max(target_ll, 1)
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())


        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc,input.size(0))
        da_IoU_seg.update(da_IoU,input.size(0))
        da_mIoU_seg.update(da_mIoU,input.size(0))


        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())


        ll_acc = LL.pixelAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()

        ll_acc_seg.update(ll_acc,input.size(0))
        ll_IoU_seg.update(ll_IoU,input.size(0))
        ll_mIoU_seg.update(ll_mIoU,input.size(0))

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)

    print("DA :",da_segment_result)
    print("LL :",ll_segment_result)



def validation(args):
    model=Model()
    if not os.path.isfile(args.cached_data_file):
        dataLoad = ld.LoadData(args.data_dir, args.classes, args.cached_data_file)
        
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'],valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val(valLoader, model.fuse())



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNetv2", help='Model name')
    parser.add_argument('--data_dir', default="/home/ceec/huycq/data/esp", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=1024, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--scaleIn', type=int, default=1, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
    parser.add_argument('--max_epochs', type=int, default=300, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./results_espnetv2_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')  #
    parser.add_argument('--classes', type=int, default=3, help='No of classes in the dataset. 20 for cityscapes')
    parser.add_argument('--cached_data_file', default='city.p', help='Cached file name')
    parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--s', default=1, type=float, help='scaling parameter')

    validation(parser.parse_args())
