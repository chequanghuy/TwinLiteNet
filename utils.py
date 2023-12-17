
import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
import yaml
import matplotlib
import matplotlib.pyplot as plt


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.2, 0.8]  # weights for [da_mIoU_seg, ll_IoU_seg]
    return (x[:, :2] * w).sum(1)

def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml'):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (da_acc_seg, da_IoU_seg, da_mIoU_seg, ll_acc_seg, ll_IoU_seg, ll_mIoU_seg)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 2])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :2])
        c = '%10.4g' * len(results) % results  # results (da_acc_seg, da_IoU_seg, da_mIoU_seg, ll_acc_seg, ll_IoU_seg, ll_mIoU_seg)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

def plot_evolution(yaml_file='data/hyp.finetune.yaml'):  # from utils.plots import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)

    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 2]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')

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

def poly_lr_scheduler(args, hyp, optimizer, epoch, power=1.5):
    lr = round(hyp['lr'] * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def train(args, train_loader, model, criterion, optimizer, epoch,scaler,verbose=False):
    model.train()
    print("epoch: ", epoch)
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    if verbose:
        LOGGER.info(('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss'))
        pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}')
    for i, (_,input, target) in pbar:
        optimizer.zero_grad()
        if args.onGPU == True:
            input = input.cuda().float() / 255.0        
        output = model(input)
        with torch.cuda.amp.autocast():
            focal_loss,tversky_loss,loss = criterion(output,target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if verbose:
            pbar.set_description(('%13s' * 1 + '%13.4g' * 3) %
                                     (f'{epoch}/{300 - 1}', tversky_loss, focal_loss, loss.item()))




@torch.no_grad()
def val(val_loader = None, model = None, half = False, is320=False, args=None):

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
    if args.verbose:
        pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.cuda().half() / 255.0 if half else input.cuda().float() / 255.0
        
        input_var = input
        target_var = target

        # run the mdoel
        with torch.no_grad():
            output = model(input_var)


        if not args.sell:
            ###-------------Drivable Segmetation--------------
            if args.seda:
                out_da = output
            else:
                out_da = output[0]

            target_da = target[0]
            if is320:
                out_da=torch.nn.functional.interpolate(out_da, scale_factor=2, mode='bilinear', align_corners=True)

            _,da_predict = torch.max(out_da, 1)
            da_predict = da_predict[:,12:-12]
            _,da_gt=torch.max(target_da, 1)

            DA.reset()
            DA.addBatch(da_predict.cpu(), da_gt.cpu())

            da_acc = DA.pixelAccuracy()
            da_IoU = DA.IntersectionOverUnion()
            da_mIoU = DA.meanIntersectionOverUnion()

            da_acc_seg.update(da_acc,input.size(0))
            da_IoU_seg.update(da_IoU,input.size(0))
            da_mIoU_seg.update(da_mIoU,input.size(0))

            ###-------------Drivable Segmetation--------------

        if not args.seda:
            ###-------------Lane Segmetation-----------------
            if args.sell:
                out_ll = output
            else:
                out_ll = output[1]
            target_ll = target[1]
            if is320:
                out_ll=torch.nn.functional.interpolate(out_ll, scale_factor=2, mode='bilinear', align_corners=True)

            _,ll_predict=torch.max(out_ll, 1)
            ll_predict = ll_predict[:,12:-12]
            _,ll_gt=torch.max(target_ll, 1)
            
            LL.reset()
            LL.addBatch(ll_predict.cpu(), ll_gt.cpu())

            ll_acc = LL.lineAccuracy()
            ll_IoU = LL.IntersectionOverUnion()
            ll_mIoU = LL.meanIntersectionOverUnion()

            ll_acc_seg.update(ll_acc,input.size(0))
            ll_IoU_seg.update(ll_IoU,input.size(0))
            ll_mIoU_seg.update(ll_mIoU,input.size(0))

            ###-------------Lane Segmetation-----------------
    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)
    
    return da_segment_result,ll_segment_result


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])