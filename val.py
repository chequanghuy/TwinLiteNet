import torch
import torch
from model import TwinLite2 as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import val, netParams
import torch.optim.lr_scheduler
from const import *
from loss import TotalLoss

import numpy as np
import time
import random
import yaml
from pathlib import Path


def validation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    model = net.TwinLiteNet(args)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = model.cuda()
        cudnn.benchmark = True
        
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  
    if not args.is320:
        valLoader = torch.utils.data.DataLoader(
            myDataLoader.Dataset(hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], hyp["hgain"], hyp["sgain"], hyp["vgain"], valid=True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        example = torch.rand(args.batch_size, 3, 384, 640).cuda().half() if args.half else torch.rand(args.batch_size, 3, 384, 640).cuda()
    else:
        valLoader = torch.utils.data.DataLoader(
            myDataLoader.Dataset320(hyp["degrees"], hyp["translate"], hyp["scale"], hyp["shear"], hyp["hgain"], hyp["sgain"], hyp["vgain"], valid=True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        example = torch.rand(args.batch_size, 3, 192, 320).cuda().half() if args.half else torch.rand(args.batch_size, 3, 192, 320).cuda()

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    model = torch.jit.trace(model.half() if args.half else model, example)
    if args.seda:
        da_segment_results = val(valLoader, model, args.half, is320 = args.is320, args = args)[0]

        msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})'.format(
                            da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2])
    elif args.sell:
        ll_segment_results = val(valLoader, model, args.half, is320 = args.is320, args = args)[1]

        msg =  'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                            ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    else:
        da_segment_results,ll_segment_results = val(valLoader, model, args.half, is320 = args.is320, args = args)

        msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                        'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                            da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                            ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    
    print(msg)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--weight', default="")
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--type', help='')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='hyperparameters path')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--is320', action='store_true')
    parser.add_argument('--seda', action='store_true', help='sigle encoder for Driable Segmentation')
    parser.add_argument('--sell', action='store_true', help='sigle encoder for Lane Segmentation')
    

    validation(parser.parse_args())
