import sys
sys.path.append('/home/ceec/huycq/HybridNets')
import torch
import numpy as np
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
def Visualize(model):
    crf2=CRF(2)
    image_list=os.listdir('/home/ceec/huycq/data/bdd100k/images/val')
    shutil.rmtree('/home/ceec/huycq/ESPNetv2/segmentation/visualize_re')
    os.mkdir('visualize_re')
    for i, imgName in enumerate(image_list[60:100]):
        img = cv2.imread(os.path.join('/home/ceec/huycq/data/bdd100k/images/val',imgName))
        img = cv2.resize(img, (640, 360))
        img_rs=img.copy()
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img=torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)  # add a batch dimension
        print(img.size())
        
        img=img.cuda().float() / 255.0
        img_variable = img
        img_variable = img_variable.cuda()
        # for i in range(10):
        st=time.time()
        with torch.no_grad():
            img_out = model(img_variable)
        print(time.time()-st)

        _,da_predict=torch.max(crf2(crf2(img_out[0])), 1)
        _,ll_predict=torch.max(crf2(crf2(img_out[1])), 1)

        DA = da_predict.byte().cpu().data.numpy()[0]*255
        LL = ll_predict.byte().cpu().data.numpy()[0]*255

        # DA = cv2.resize(DA, (1280, 720), interpolation = cv2.INTER_LINEAR)

        # LL = cv2.resize(LL, (1280, 720), interpolation = cv2.INTER_LINEAR)
        img_rs[DA>100]=[255,0,0]
        img_rs[LL>100]=[0,255,0]        
        cv2.imwrite(os.path.join('/home/ceec/huycq/ESPNetv2/segmentation/visualize_re',imgName),img_rs)
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
@torch.no_grad()
def val1(val_loader, model):
    seg_list=["DA","LL"]
    ncs = 3
    seen = 0
    s_seg = ' ' * (15 + 11 * 8)
    s=""
    for i in range(len(seg_list)):
        s_seg += '%-33s' % seg_list[i]
        s += ('%-11s' * 3) % ('mIoU', 'IoU', 'Acc')
    iou_ls = [[] for _ in range(ncs)]
    acc_ls = [[] for _ in range(ncs)]

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    crf2=CRF(2)
    for i, (image_name,input, target) in pbar:
        target=target.cuda()
        input = input.cuda().float() / 255.0
            # target = target.cuda()

        input_var = input
        # run the mdoel
        import time
        start=time.time()
        with torch.no_grad():
            segmentation = model(input_var)
        # print(time.time()-start)
        import cv2
        segmentation = segmentation.log_softmax(dim=1).exp()
        _, segmentation = torch.max(crf2(crf2(segmentation)), 1)  # (bs, C, H, W) -> (bs, H, W)
        for i in range(segmentation.size(0)):
            # print(image_name,type(image_name))
            img_vs=cv2.imread(image_name[i])
            
            img_vs=cv2.resize(img_vs,(640,384))
            seg_mask_ = segmentation[i].byte().cpu().data.numpy()
            # print(seg_mask_)
            # seg_mask_ = cv2.cvtColor(seg_mask_,cv2.COLOR_GRAY2BGR)
            img_vs[seg_mask_==1]=[255,0,0]
            img_vs[seg_mask_==2]=[255,255,0]
            # cv2.imwrite("re/{}.jpg".format(i),seg_mask_*0.5+img_vs*0.5)
            cv2.imwrite("re/{}.jpg".format(i),img_vs)



        _,target= torch.max(target, 1)
        
        tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation, target, mode="multiclass",
                                                                threshold=None,
                                                                num_classes=ncs)
        iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
        #         print(iou)
        acc = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
        # print(iou)
        for i in range(ncs):
            iou_ls[i].append(iou.T[i].detach().cpu().numpy())
            acc_ls[i].append(acc.T[i].detach().cpu().numpy())

        


    for i in range(ncs):
        iou_ls[i] = np.concatenate(iou_ls[i])
        acc_ls[i] = np.concatenate(acc_ls[i])
    # print(len(iou_ls[0]))
    iou_score = np.mean(iou_ls)
    # print(iou_score)
    acc_score = np.mean(acc_ls)

    miou_ls = []
    for i in range(len(seg_list)):
        miou_ls.append(np.mean( (iou_ls[0] + iou_ls[i+1]) / 2))

    for i in range(ncs):
        iou_ls[i] = np.mean(iou_ls[i])
        acc_ls[i] = np.mean(acc_ls[i])


    print(s_seg)
    print(s)
    pf = ('%-11.3g' * 2) % (iou_score, acc_score)
    for i in range(len(seg_list)):
        tmp = i+1
        pf += ('%-11.3g' * 3) % (miou_ls[i], iou_ls[tmp], acc_ls[tmp])
    print(pf)



def validation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    model = net.ESPNet()

    if num_gpus >= 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + str(args.s) + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    # check if processed data file exists or not
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

    if cuda_available:
        args.onGPU = True
        model = model.cuda()

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # define optimization criteria
    



    if args.onGPU:
        cudnn.benchmark = True
    # checkpoint = torch.load(args.resume)
    model.load_state_dict(torch.load('/home/ceec/huycq/ESPNetv2/segmentation/cus_1/model_92.pth'))
    model.eval()
    print("load done")
    # Visualize(model)
    # val(valLoader, model)

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
