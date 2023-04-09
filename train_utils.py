#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================
from IOUEval import iouEval
import time
import torch
import numpy as np
from IOUEval import iouEval,SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import argparse
from argparse import ArgumentParser
from utils import smp_metrics
from utils.constants import *
# import sys
# sys.append('/home/ceec/huycq/FastestDet/module/')
LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

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

def poly_lr_scheduler(args, optimizer, epoch, power=2):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr



def train(args, train_loader, model, criterion, optimizer, epoch):
    model.train()
    epoch_loss = [0.1,0.3]

    total_batches = len(train_loader)
    # pbar = enumerate(train_loader)
    # LOGGER.info(('\n' + '%11s' * 4) % ('Epoch','LOSS1','LOSS2' ,'LOSS'))
    # pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in enumerate(train_loader):
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        
        # target=target.cuda()
        optimizer.zero_grad()

        loss1,loss2,loss = criterion(output,target)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # pbar.set_description(('%11s' * 1 + '%11.4g' * 3) %
        #                              (f'{epoch}/{300 - 1}', loss1, loss2, loss.item()))




    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train

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




# @torch.no_grad()
# def val(val_loader, model):
#     seg_list=["DA","LL"]
#     ncs = 3
#     seen = 0
#     s_seg = ' ' * (15 + 11 * 8)
#     s=""
#     for i in range(len(seg_list)):
#         s_seg += '%-33s' % seg_list[i]
#         s += ('%-11s' * 3) % ('mIoU', 'IoU', 'Acc')
#     iou_ls = [[] for _ in range(ncs)]
#     acc_ls = [[] for _ in range(ncs)]

#     total_batches = len(val_loader)
#     pbar = enumerate(val_loader)
#     pbar = tqdm(pbar, total=total_batches)
#     for i, (image_name,input, target) in pbar:
#         target=target.cuda()
#         input = input.cuda().float() / 255.0

#         input_var = input
#         with torch.no_grad():
#             segmentation = model(input_var)

#         segmentation = segmentation.log_softmax(dim=1).exp()
#         _, segmentation = torch.max(segmentation, 1)  # (bs, C, H, W) -> (bs, H, W)
#         _,target= torch.max(target, 1)
        
#         tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation, target, mode="multiclass",
#                                                                 threshold=None,
#                                                                 num_classes=ncs)
#         iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
#         #         print(iou)
#         acc = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
#         # print(iou)
#         for i in range(ncs):
#             iou_ls[i].append(iou.T[i].detach().cpu().numpy())
#             acc_ls[i].append(acc.T[i].detach().cpu().numpy())

        


#     for i in range(ncs):
#         iou_ls[i] = np.concatenate(iou_ls[i])
#         acc_ls[i] = np.concatenate(acc_ls[i])
#     # print(len(iou_ls[0]))
#     iou_score = np.mean(iou_ls)
#     # print(iou_score)
#     acc_score = np.mean(acc_ls)

#     miou_ls = []
#     for i in range(len(seg_list)):
#         miou_ls.append(np.mean( (iou_ls[0] + iou_ls[i+1]) / 2))

#     for i in range(ncs):
#         iou_ls[i] = np.mean(iou_ls[i])
#         acc_ls[i] = np.mean(acc_ls[i])


#     print(s_seg)
#     print(s)
#     pf = ('%-11.3g' * 2) % (iou_score, acc_score)
#     for i in range(len(seg_list)):
#         tmp = i+1
#         pf += ('%-11.3g' * 3) % (miou_ls[i], iou_ls[tmp], acc_ls[tmp])
#     print(pf)

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])