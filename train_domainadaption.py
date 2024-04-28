import os
import sys 
# put the directory efficientvit instead of '..'
sys.path.insert(1, os.path.join(sys.path[0], '/kaggle/working/efficientvit'))
######
import torch
import pickle
from model import TwinLite as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from torchvision.transforms import transforms as T
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from DataSet import MyDataset,IADDataset,MIXEDataset,BDDataset,first_pseudo_label_dataset

if not os.path.isdir('/kaggle/working/iadd/ll'):
    os.mkdir('/kaggle/working/iadd/ll')
    print('making ll folder')
if not os.path.isdir('/kaggle/working/iadd/da'):
    os.mkdir('/kaggle/working/iadd/da')
    print('making da folder')

transform=T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    ),

])

from efficientvit.seg_model_zoo import create_seg_model

from loss import TotalLoss
import os

path_list=[]
# for root, dirs, files in os.walk('/kaggle/working/iadd/img'):
for root, dirs, files in os.walk('/kaggle/input/bdd100k-dataset/bdd100k/bdd100k/images/100k/train'):
  for name in files:
    path=os.path.join(root,name)
    if path[-4:]=='.jpg':
      path_list.append(path)
path_list=path_list[:10000]
def resize(
    x: torch.Tensor,
    size: any or None = None,
    scale_factor: list[float] or None = None,
    mode: str = "bicubic",
    align_corners: bool or None = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")
class SegmentationMetric(object):
    '''
    imgLabel [batch_size, height(144), width(256)]
    confusionMatrix [[0(TN),1(FP)],
                     [2(FN),3(TP)]]
    '''
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
        
    def lineAccuracy(self):
        Acc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-12)
        return Acc[1]

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-12)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)
        return mIoU
    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        # print(imgLabel.shape)
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
        
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

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        # print(imgLabel.shape)
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
def save_res(i,vis_idx, input, out_da, out_ll, target_da, target_ll):
    
    
    x=input
    seg=target_da
    ll=target_ll

    y_seg_pred=out_da
    y_ll_pred=out_ll

    vis_pred1=(y_seg_pred)[vis_idx][1].detach().cpu().numpy()
    vis_pred2=(y_seg_pred)[vis_idx][0].detach().cpu().numpy()
    vis_pred3=(y_ll_pred)[vis_idx][1].detach().cpu().numpy()
    vis_pred4=(y_ll_pred)[vis_idx][0].detach().cpu().numpy()

    vis_logit=(y_seg_pred)[vis_idx].argmax(0).detach().cpu().numpy()
    vis_logit2=(y_ll_pred)[vis_idx].argmax(0).detach().cpu().numpy()

    vis_input=invTrans(x[vis_idx]).permute(1,2,0).cpu().numpy()
    vis_input = cv2.cvtColor(vis_input, cv2.COLOR_BGR2RGB)


    vis_label1= seg[vis_idx][1].long().detach().cpu().numpy()
    vis_label2= ll[vis_idx][1].long().detach().cpu().numpy()

    viss = [vis_pred1,vis_pred2,vis_pred3,vis_pred4,vis_logit,vis_logit2,vis_label1,vis_label2,vis_input]
#     show_grays(viss,3)

    img_det1 = show_seg_result(vis_input*255, (vis_logit, vis_logit2), 0, 0, is_demo=True)
    img_det2 = show_seg_result(vis_input*255, (vis_label1, vis_label2), 0, 0, is_demo=True)


    filename1 = f'savedImage1_{i}_{vis_idx}.jpg'
    filename2 = f'savedImage2_{i}_{vis_idx}.jpg'

    img_det1 = cv2.cvtColor(img_det1, cv2.COLOR_RGB2BGR)
    img_det2 = cv2.cvtColor(img_det2, cv2.COLOR_RGB2BGR)

    # Using cv2.imwrite() method 
    # Saving the image 
    cv2.imwrite(f'/kaggle/working/outputs/{filename1}', img_det1) 
    cv2.imwrite(f'/kaggle/working/outputs/{filename2}', img_det2) 

@torch.no_grad()
def val(val_loader, model):
    
    os.mkdir('/kaggle/working/outputs')
    
    model.eval()


    DA=SegmentationMetric(2)
    LL=SegmentationMetric(2)

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
        input = input.cuda().float()
            # target = target.cuda()

        input_var = input
        target_var = target
#         print(target_var[2].shape)

        # run the mdoel
        with torch.no_grad():
            output = model(input_var)
            output = (resize(output[0],[512, 512]), resize(output[1],[512, 512]))

#         print(output[0].shape)
        
        out_da,out_ll=output
        target_da,target_ll=target
        
         
            
        
#         print('da_gt',da_gt.shape)

#         target_da = resize(target_da,[64, 64])
#         target_ll = resize(target_ll,[64, 64])
    
#         print('target_da',target_da.shape)
        
        _,da_gt=torch.max(target_da, 1)
        _,da_predict=torch.max(out_da, 1)
        
#         print('da_gt',da_gt.shape)
#         print('da_predict',da_predict.shape)
        
        _,ll_predict=torch.max(out_ll, 1)
        _,ll_gt=torch.max(target_ll, 1)
#         ll_gt = resize(ll_gt,[64, 64])[0]
        
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
        
#         if printed : 
#         save_res(i,0, input, out_da, out_ll, target_da, target_ll)
            
#         for indx in range(32):
#             if indx%4 ==0:
#                 save_res(i,vis_idx, input, out_da, out_ll, target_da, target_ll)

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    return da_segment_result,ll_segment_result
    
def valid(mymodel,Dataset):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    # load the model
    model = mymodel.eval()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True


    valLoader = torch.utils.data.DataLoader(
        Dataset,
        batch_size=2, shuffle=False, num_workers=1, pin_memory=True)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

#     model.load_state_dict(torch.load(PATH))
    model.eval()
#     example = torch.rand(2, 3, 512, 512).cuda()
#     model = torch.jit.trace(model, example)
    da_segment_results , ll_segment_results = val(valLoader, model)

    msg =  '\n Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    print(msg)
    
    msg2 =  '\n lane line detection: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})    mIOU({ll_seg_miou:.3f})\n'.format(
                          da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    print(msg2)
    
def pseudo_label_maker(dataloader,model):
    # model = create_seg_model('b0','bdd',weight_url='/kaggle/working/model_0.pth')
    model=model.cuda()
    model.eval()
    tbar = tqdm(dataloader)
#     loop  = tqdm(names)

#     bch=iter(pseudo_data)
    with torch.no_grad():
        for name, image , shape in tbar:
#             print(name)
    #         image=cv2.imread(name)
    #         img = image.astype(np.uint8)
    #         img = cv2.resize(img, [512,512], interpolation=cv2.INTER_LINEAR)
    #         img=transform(img).unsqueeze(0).cuda()
            y_da_pred , y_ll_pred=model(image.cuda())
            
            H , W = shape[0] , shape[1]
            y_da_pred=resize(y_da_pred, [H , W])
            y_ll_pred=resize(y_ll_pred, [H , W])


            y_da_pred=y_da_pred[0].argmax(0).detach().cpu().numpy()
            y_ll_pred=y_ll_pred[0].argmax(0).detach().cpu().numpy()

            y_da_pred=y_da_pred.astype(np.uint8)*255
            y_ll_pred=y_ll_pred.astype(np.uint8)*255

            nam=name[0].split('/')[-1]
            da_name='/kaggle/working/iadd/da/' + nam.replace('.jpg','.png')
            ll_name='/kaggle/working/iadd/ll/' + nam.replace('.jpg','.png')

            cv2.imwrite(da_name,y_da_pred)
            cv2.imwrite(ll_name,y_ll_pred)
            tbar.set_description('pseudo relabeling: ')


def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    # model = net.TwinLiteNet()
    transform=T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
            ),
    
        ])    
    pretrained=args.pretrained
    if pretrained is not None:
        model = create_seg_model('b0','bdd',weight_url=pretrained)
        # if args.pseudo:
            # pseudo_data = torch.utils.data.DataLoader(
            #     first_pseudo_label_dataset(transform=transform, valid=False),
            #     batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
            # print(' pseudo label makering using the pretrained weights')
            # pseudo_label_maker(pseudo_data,model)
    else:
        model = create_seg_model('b0','bdd',False)

    # pseudo_label_maker(path_list,model)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    
    # trainLoader = torch.utils.data.DataLoader(
    #     myDataLoader.MIXEDataset(transform=transform,valid=False),
    #     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    # valLoader = torch.utils.data.DataLoader(
    #     myDataLoader.MIXEDataset(valid=True),
    #     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    trainLoader = torch.utils.data.DataLoader(
        MIXEDataset(transform=transform,valid=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    valLoader = torch.utils.data.DataLoader(
        MIXEDataset(valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True
        
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.head1.parameters():
    #     param.requires_grad = True   
        
    trainables=0
    for param in model.parameters():
        if param.requires_grad == True:
           trainables+=1
            
        param.requires_grad = True
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    print('Trainable network parameters: ' + str(trainables))

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            lr=checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            if args.pseudo:
                pseudo_data = torch.utils.data.DataLoader(
                    first_pseudo_label_dataset(transform=transform, valid=False),
                    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
                pseudo_label_maker(pseudo_data,model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, 0))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " +  str(lr))

        # train for one epoch
        model.train()
        train( args, trainLoader, model, criteria, optimizer, epoch)
        # model.eval()
        # # validation
        # da_segment_results , ll_segment_results = val(valLoader, model)
        
        torch.save(model.state_dict(), model_file_name)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        # Dataset0= MIXEDataset(transform=transform , valid=True)
        pseudo_data = torch.utils.data.DataLoader(
            first_pseudo_label_dataset(transform=transform, valid=False),
            batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        valid(model,MIXEDataset(transform=transform , valid=True))
        pseudo_label_maker(pseudo_data,model)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-5 , help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default=None, help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--pseudo', default=True, help='Pretrained ESPNetv2 weights.')

    train_net(parser.parse_args())
