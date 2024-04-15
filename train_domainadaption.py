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
from torchvision.transforms import transforms as T

transform2=T.Compose([
#     T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    ),

])

from efficientvit.seg_model_zoo import create_seg_model

from loss import TotalLoss
import os

path_list=[]
for root, dirs, files in os.walk('/kaggle/working/iadd/img'):
  for name in files:
    path=os.path.join(root,name)
    if path[-4:]=='.jpg':
      path_list.append(path)
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
        
def pseudo_label_maker(names,model):
    # model = create_seg_model('b0','bdd',weight_url='/kaggle/input/model149/model_149.pth')
    model=model.cuda()
    convert_tensor = T.ToTensor()
    for name in names:
        image=cv2.imread(name)
        img = image.astype(np.uint8)
        img = cv2.resize(img, [512,512], interpolation=cv2.INTER_LINEAR)
        img=convert_tensor(img).unsqueeze(0).cuda()
        y_da_pred , y_ll_pred=model(transform2(img))
        
        y_da_pred=resize(y_da_pred, [1080, 1920])
        y_ll_pred=resize(y_ll_pred, [1080, 1920])
        
        y_da_pred=y_da_pred[0].argmax(0).detach().cpu().numpy()
        y_ll_pred=y_ll_pred[0].argmax(0).detach().cpu().numpy()
        
        y_da_pred=y_da_pred.astype(np.uint8)
        y_ll_pred=y_ll_pred.astype(np.uint8)
        
        nam=name.split('/')[-1]
        da_name='/kaggle/working/iadd/da/' + nam.replace('.jpg','.png')
        ll_name='/kaggle/working/iadd/ll/' + nam.replace('.jpg','.png')
        
        cv2.imwrite(da_name,y_da_pred)
        cv2.imwrite(ll_name,y_ll_pred)

model = create_seg_model('b0','bdd',weight_url='/kaggle/working/TwinLiteNet/model/model_149.pth')
pseudo_label_maker(path_list,model)
def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    # model = net.TwinLiteNet()
    model = create_seg_model('b0','bdd',weight_url='/kaggle/working/model_0.pth')
    # pseudo_label_maker(path_list,model)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    transform=T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    ),

])
    
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MIXEDataset(transform=transform,valid=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MIXEDataset(valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = 0
            lr=args.lr
            model.load_state_dict(checkpoint['state_dict'])
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

        pseudo_label_maker(path_list,model)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default='model/model_149.pth', help='Pretrained ESPNetv2 weights.')

    train_net(parser.parse_args())
