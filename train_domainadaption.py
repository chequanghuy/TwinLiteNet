import os
import sys

# put the directory efficientvit instead of '..'

######
import torch
import pickle
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from utils import train, valid, netParams, save_checkpoint, poly_lr_scheduler, pseudo_label_maker
import torch.optim.lr_scheduler
from torchvision.transforms import transforms as T
import DataSet as myDataLoader
from loss import TotalLoss, DiscriminatorLoss, MMDLoss, MMDTotal
import os
import torch.backends.cudnn as cudnn
from model.Discriminator import FCDiscriminator
import torch.nn.functional as F
import torch.nn as nn


def train_net(args):
    if args.engine == 'kaggle':
        sys.path.insert(1, os.path.join(sys.path[0], '/kaggle/working/efficientvit'))
        from efficientvit.seg_model_zoo import create_seg_model
    else:
        sys.path.insert(1, os.path.join(sys.path[0], '/content/efficientvit'))
        from efficientvit.seg_model_zoo import create_seg_model

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
    model_D = FCDiscriminator(num_classes=128)


    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        model_D = model_D.cuda()
        cudnn.benchmark = True

    criteria = TotalLoss(device=args.device)
    criteria_bce = torch.nn.MSELoss().to(args.device)
    start_epoch = 0
    lr = args.lr
    lr_D = args.lr / 4

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=5e-4)

    optimizer.zero_grad()
    optimizer_D.zero_grad()

    input_size = [512, 512]
    input_size_target = [512, 512]


    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

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

    iadd_valLoader = myDataLoader.MyDataset(transform=transform, valid=True, engin=engine, data='IADD')

    bdd_valLoader = myDataLoader.MyDataset(transform=transform, valid=True, engin=engine, data='bdd')

    source_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(transform=transform, valid=False, engin=engine, data='bdd'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    target_loader = torch.utils.data.DataLoader(
        myDataLoader.UlabeledDataset(transform=transform, engin=engine, data='IADD'),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        checkpoint_file_name = args.savedir + os.sep + 'checkpoint_{}.pth.tar'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        poly_lr_scheduler(args, optimizer_D, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))
        # train for one epoch
        train(args, source_loader, target_loader, model, model_D, criteria, criteria_bce, optimizer, optimizer_D, epoch)

        # valid(model, iadd_valLoader)
        # valid(model, bdd_valLoader)

        torch.save(model.state_dict(), model_file_name)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, checkpoint_file_name)

        valid(model, iadd_valLoader)
        valid(model, bdd_valLoader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
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
