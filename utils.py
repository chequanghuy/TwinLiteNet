import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
import torch.nn.functional as F
import cv2
import torch
import pickle
from model import TwinLite as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
import torch.optim.lr_scheduler
from torchvision.transforms import transforms as T
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from itertools import cycle


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


LOGGING_NAME = "custom"


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
                'level': level, }},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False, }}})


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

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        # print(imgLabel.shape)
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix


def poly_lr_scheduler(args, optimizer, epoch, power=2):
    lr = round(args.lr * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train(args, source_loader, target_loader, model, criterion, criterion_mmd, optimizer, epoch):
    model.train()
    # disc_model.train()

    total_batches = len(source_loader)
    target_loader = cycle(target_loader)
    # pbar = enumerate(zip(source_loader, cycle(target_loader)))
    LOGGER.info(('\n' + '%13s' * 5) % ('Epoch', 'TverskyLoss', 'FocalLoss', 'MMDLoss', 'TotalLoss'))
    # pbar = tqdm(pbar, total=total_batches, )
    pbar = tqdm(iter(source_loader), bar_format='{l_bar}{bar:10}{r_bar}')
    for i in pbar:
        (_, source_input, source_label) = next(source_loader)
        (_, target_input, _) = next(target_loader)
        if args.device == 'cuda:0':
            source_input = source_input.cuda().float()
            source_label[0] = source_label[0].cuda()
            source_label[1] = source_label[1].cuda()
            target_input = target_input.cuda().float()
        optimizer.zero_grad()
        source_output = model(source_input)
        target_output = model(target_input)

        mmd_loss = criterion_mmd(source_output, target_output)

        output_source = (resize(source_output[0], [512, 512]), resize(source_output[1], [512, 512]))

        # Source features labeled as 1, target features labeled as 0
        # discriminator_input = torch.cat([source_features, target_features], dim=0)
        # discriminator_labels = torch.cat([torch.ones(source_features.size(0)), torch.zeros(target_features.size(0))],dim=0).to(device)
        # discriminator_output = disc_model(discriminator_input)
        # disc_loss = criterion_discriminator(discriminator_output.squeeze(), discriminator_labels)
        # disc_loss = F.binary_cross_entropy(discriminator_output.squeeze(), discriminator_labels)
        # disc_loss.backward()
        # disc_optimizer.step()
        # disc_optimizer.zero_grad()

        # Train segmentation model to fool the discriminator
        # target_features = target_output.view(target_output.size(0), -1)
        # discriminator_output = disc_model(target_features)
        # adversarial_loss = F.binary_cross_entropy(discriminator_output.squeeze(),torch.ones(target_features.size(0)).to(device))
        # adversarial_loss = criterion_adversarial(discriminator_output.squeeze(),torch.ones(target_features.size(0)).to(device))
        focal_loss, tversky_loss, loss = criterion(output_source, source_label)

        total_loss = loss + mmd_loss
        total_loss.backward()
        optimizer.step()

        pbar.set_description(('%13s' * 1 + '%13.4g' * 4) %
                             (f'{epoch}/{args.max_epochs - 1}', tversky_loss, focal_loss, mmd_loss, total_loss.item()))


def train16fp(args, train_loader, model, criterion, optimizer, epoch, scaler):
    model.train()
    print("16fp-------------------")
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    LOGGER.info(('\n' + '%13s' * 4) % ('Epoch', 'TverskyLoss', 'FocalLoss', 'TotalLoss'))
    pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_, input, target) in pbar:
        optimizer.zero_grad()
        if args.onGPU == True:
            input = input.cuda().float()
        output = model(input)
        with torch.cuda.amp.autocast():
            focal_loss, tversky_loss, loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                             (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))


@torch.no_grad()
def val(val_loader, model):
    # os.mkdir('/kaggle/working/outputs')

    model.eval()

    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    pbar = tqdm(pbar, total=total_batches)
    for i, (_, input, target) in pbar:
        input = input.cuda().float()
        # target = target.cuda()

        input_var = input
        target_var = target

        with torch.no_grad():
            output = model(input_var)
            output = (resize(output[0], [512, 512]), resize(output[1], [512, 512]))

        out_da, out_ll = output
        target_da, target_ll = target

        _, da_gt = torch.max(target_da, 1)
        _, da_predict = torch.max(out_da, 1)

        _, ll_predict = torch.max(out_ll, 1)
        _, ll_gt = torch.max(target_ll, 1)

        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())

        da_acc = DA.pixelAccuracy()
        da_IoU = DA.IntersectionOverUnion()
        da_mIoU = DA.meanIntersectionOverUnion()

        da_acc_seg.update(da_acc, input.size(0))
        da_IoU_seg.update(da_IoU, input.size(0))
        da_mIoU_seg.update(da_mIoU, input.size(0))

        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())

        ll_acc = LL.pixelAccuracy()
        ll_IoU = LL.IntersectionOverUnion()
        ll_mIoU = LL.meanIntersectionOverUnion()

        ll_acc_seg.update(ll_acc, input.size(0))
        ll_IoU_seg.update(ll_IoU, input.size(0))
        ll_mIoU_seg.update(ll_mIoU, input.size(0))

    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)
    return da_segment_result, ll_segment_result


def valid(mymodel, Dataset):
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
    da_segment_results, ll_segment_results = val(valLoader, model)

    msg = '\n Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n'.format(
        da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
        ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2])
    print(msg)

    msg2 = '\n lane line detection: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})    mIOU({ll_seg_miou:.3f})\n'.format(
        da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
        ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2])
    print(msg2)


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)


def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])


def save_res(i, vis_idx, input, out_da, out_ll, target_da, target_ll):
    x = input
    seg = target_da
    ll = target_ll

    y_seg_pred = out_da
    y_ll_pred = out_ll

    vis_pred1 = (y_seg_pred)[vis_idx][1].detach().cpu().numpy()
    vis_pred2 = (y_seg_pred)[vis_idx][0].detach().cpu().numpy()
    vis_pred3 = (y_ll_pred)[vis_idx][1].detach().cpu().numpy()
    vis_pred4 = (y_ll_pred)[vis_idx][0].detach().cpu().numpy()

    vis_logit = (y_seg_pred)[vis_idx].argmax(0).detach().cpu().numpy()
    vis_logit2 = (y_ll_pred)[vis_idx].argmax(0).detach().cpu().numpy()

    vis_input = invTrans(x[vis_idx]).permute(1, 2, 0).cpu().numpy()
    vis_input = cv2.cvtColor(vis_input, cv2.COLOR_BGR2RGB)

    vis_label1 = seg[vis_idx][1].long().detach().cpu().numpy()
    vis_label2 = ll[vis_idx][1].long().detach().cpu().numpy()

    viss = [vis_pred1, vis_pred2, vis_pred3, vis_pred4, vis_logit, vis_logit2, vis_label1, vis_label2, vis_input]
    #     show_grays(viss,3)

    img_det1 = show_seg_result(vis_input * 255, (vis_logit, vis_logit2), 0, 0, is_demo=True)
    img_det2 = show_seg_result(vis_input * 255, (vis_label1, vis_label2), 0, 0, is_demo=True)

    filename1 = f'savedImage1_{i}_{vis_idx}.jpg'
    filename2 = f'savedImage2_{i}_{vis_idx}.jpg'

    img_det1 = cv2.cvtColor(img_det1, cv2.COLOR_RGB2BGR)
    img_det2 = cv2.cvtColor(img_det2, cv2.COLOR_RGB2BGR)

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(f'/kaggle/working/outputs/{filename1}', img_det1)
    cv2.imwrite(f'/kaggle/working/outputs/{filename2}', img_det2)


def pseudo_label_maker(dataloader, model):
    # model = create_seg_model('b0','bdd',weight_url='/kaggle/working/model_0.pth')
    if not os.path.isdir('/kaggle/working/iadd/ll'):
        os.mkdir('/kaggle/working/iadd/ll')
        print('making ll folder')
    if not os.path.isdir('/kaggle/working/iadd/da'):
        os.mkdir('/kaggle/working/iadd/da')
        print('making da folder')

    model = model.cuda()
    model.eval()
    tbar = tqdm(dataloader)
    #     loop  = tqdm(names)

    #     bch=iter(pseudo_data)
    with torch.no_grad():
        for name, image, shape in tbar:
            #             print(name)
            #         image=cv2.imread(name)
            #         img = image.astype(np.uint8)
            #         img = cv2.resize(img, [512,512], interpolation=cv2.INTER_LINEAR)
            #         img=transform(img).unsqueeze(0).cuda()
            y_da_pred, y_ll_pred = model(image.cuda())

            H_, W_ = shape[0], shape[1]
            y_da_pred = resize(y_da_pred, [H_, W_])
            y_ll_pred = resize(y_ll_pred, [H_, W_])

            y_da_pred = y_da_pred[0].argmax(0).detach().cpu().numpy()
            y_ll_pred = y_ll_pred[0].argmax(0).detach().cpu().numpy()

            y_da_pred = y_da_pred.astype(np.uint8) * 255
            y_ll_pred = y_ll_pred.astype(np.uint8) * 255

            nam = name[0].split('/')[-1]
            da_name = '/kaggle/working/iadd/da/' + nam.replace('.jpg', '.png')
            ll_name = '/kaggle/working/iadd/ll/' + nam.replace('.jpg', '.png')

            cv2.imwrite(da_name, y_da_pred)
            cv2.imwrite(ll_name, y_ll_pred)
            tbar.set_description('pseudo relabeling: ')
