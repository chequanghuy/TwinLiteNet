import argparse
import torch
import torch.nn as nn
import torchvision
from utils import train16fp, val
import DataSet as myDataLoader
from torch.quantization.observer import MovingAverageMinMaxObserver
import torch
import torch
from model import TwinLite2 as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import val, val_cpu, netParams, poly_lr_scheduler
import torch.optim.lr_scheduler
from const import *
from loss import TotalLoss

import pytorch_quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import calib

import numpy as np
import time
import random
import yaml
from pathlib import Path
from tqdm import tqdm
import os
def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            # print(F"{name:40}: {module}")
    model.cuda()

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (_,image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model((image/ 255.0).cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir):


    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            # calib_output = os.path.join(
            #     out_dir,
            #     F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")
            # torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--qat', action='store_true')
    parser.add_argument('--weight', default="")
    parser.add_argument('--calibrator', default="max")
    parser.add_argument('--num_workers', type=int, default=8, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--savedir', default='', help='directory to save the results')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='hyperparameters path')
    parser.add_argument('--type', default="nano", help='')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    args = parser.parse_args()
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f) 

    calibrator = args.calibrator
    quant_modules.initialize()
    quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAdaptiveAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)

    quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
    quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], hyp["hgain"], hyp["sgain"], hyp["vgain"], valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = net.TwinLiteNet(args.type)
    model.load_state_dict(torch.load(args.weight))
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True
    
    model.eval()
    
    print("start calibration",args.qat,args.type)
    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name="nano",
            data_loader=valLoader,
            num_calib_batch=256,
            calibrator=args.calibrator,
            hist_percentile=[99.9, 99.99, 99.999, 99.9999],
            out_dir=args.savedir)
    
    
    if args.qat:
        scaler = torch.cuda.amp.GradScaler()
        criteria = TotalLoss(hyp['alpha1'], hyp['gamma1'], hyp['alpha2'], hyp['gamma2'], hyp['alpha3'], hyp['gamma3'])
        lr = hyp['lr']
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyp['lr'], betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])
        trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], hyp["hgain"], hyp["sgain"], hyp["vgain"], valid=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        model.train()
        for epoch in range(80, 90):
            poly_lr_scheduler(args,hyp,optimizer, epoch)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print(epoch," Learning rate: " +  str(lr))
            train16fp(args, trainLoader, model, criteria, optimizer, epoch,scaler,verbose=True)
        model.eval()
    print("start validation 8bit")
    da_segment_results,ll_segment_results = val(valLoader, model)
    msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    print(msg)




