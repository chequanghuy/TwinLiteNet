import os
import sys

# put the directory efficientvit instead of '..'
if args.engine == 'kaggle':
    sys.path.insert(1, os.path.join(sys.path[0], '/kaggle/working/efficientvit'))
    from efficientvit.seg_model_zoo import create_seg_model
else:
    sys.path.insert(1, os.path.join(sys.path[0], '/content/efficientvit'))
    from efficientvit.seg_model_zoo import create_seg_model

######
import torch
import pickle
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from utils import train, valid, netParams, save_checkpoint, poly_lr_scheduler, pseudo_label_maker
import torch.optim.lr_scheduler
from torchvision.transforms import transforms as T
import DataSet as myDataLoader
from loss import TotalLoss, DiscriminatorLoss,MMDLoss
import os
import torch.backends.cudnn as cudnn
from model.Discriminator import Discriminator
import torch.nn.functional as F
import  torch.nn as nn

def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    # model = net.TwinLiteNet()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

    ])
    pretrained = args.pretrained
    engine = args.engine
    if pretrained is not None:
        model = create_seg_model('b0', 'bdd', weight_url=pretrained)


    else:
        model = create_seg_model('b0', 'bdd', False)
    # disc_model =
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    criteria = TotalLoss()
    criterion_mmd = MMDLoss()
    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # disc_optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, 0))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # iadd_valLoader = torch.utils.data.DataLoader(
    iadd_valLoader=myDataLoader.MyDataset(transform=transform, valid=True, engin=engine, data='IADD')
        # batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # bdd_valLoader = torch.utils.data.DataLoader(
    bdd_valLoader=myDataLoader.MyDataset(transform=transform, valid=True, engin=engine, data='bdd')
        # batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    source_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(transform=transform, valid=False, engin=engine, data='bdd'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    target_loader = torch.utils.data.DataLoader(
        myDataLoader.UlabeledDataset(transform=transform, engin=engine,data='IADD'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # valid(model, bdd_valLoader)
    # valid(model, iadd_valLoader)

    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        # poly_lr_scheduler(args, disc_optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))
        # train for one epoch
        model = model.cuda()
        # disc_model = disc_model.cuda()
        train(args, source_loader, target_loader, model, criteria, criterion_mmd, optimizer, epoch)

        torch.save(model.state_dict(), model_file_name)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        # Dataset0= MIXEDataset(transform=transform , valid=True)
        # pseudo_data = torch.utils.data.DataLoader(
        #     first_pseudo_label_dataset2(transform=transform, valid=False),
        #     batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        valid(model, iadd_valLoader)
        valid(model, bdd_valLoader)

        # pseudo_label_maker(pseudo_data, model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate')
    parser.add_argument('--savedir', default='./test_', help='directory to save the results')
    parser.add_argument('--resume', type=str, default='', help='Use this flag to load last checkpoint for training')
    parser.add_argument('--pretrained', default=None, help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--pseudo', default=True, help='Pretrained ESPNetv2 weights.')
    parser.add_argument('--engine', default='kaggle', help='choose youre prefered engine, kaggle or colab.')

    train_net(parser.parse_args())
