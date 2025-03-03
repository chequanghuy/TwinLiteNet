import os
import sys

# put the directory efficientvit instead of '..'
sys.path.insert(1, os.path.join(sys.path[0], '/kaggle/working/efficientvit'))
######
import torch
import pickle
import copy
from model import TwinLite as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
import torch.optim.lr_scheduler
from torchvision.transforms import transforms as T

from efficientvit.seg_model_zoo import create_seg_model

from loss import TotalLoss
from efficientvit.models.efficientvit.seg import SegHead

head = SegHead(
    fid_list=["stage4", "stage3", "stage2"],
    in_channel_list=[128, 64, 32],
    stride_list=[64, 32, 16, 8],
    head_stride=4,
    head_width=32,
    head_depth=1,
    expand_ratio=4,
    middle_op="mbconv",
    final_expand=4,
    n_classes=2,
)


def train_net(args):
    # load the model
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    # model = net.TwinLiteNet()
    # model = create_seg_model('b0','bdd',False)
    if args.pretrained is not None:
        model = create_seg_model('b0', 'bdd', weight_url=args.pretrained)
    else:
        model = create_seg_model('b0', 'bdd', False)

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    args.savedir = args.savedir + '/'

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

    ])

    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    source_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    target_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if cuda_available:
        args.onGPU = True
        model = model.cuda()
        cudnn.benchmark = True

    # for param in model.parameters():
    #
    # param.requires_grad = False
    # total_paramters = netParams(model)
    #
    # for param in model.head2.parameters():
    # param.requires_grad = True
    #
    # print('Total network parameters: ' + str(total_paramters))

    criteria = TotalLoss()

    start_epoch = 0
    lr = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    # model.head1 = copy.deepcopy(head).cuda()
    # model.head2 = copy.deepcopy(head).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, args.max_epochs):

        model_file_name = args.savedir + os.sep + 'model_{}.pth'.format(epoch)
        checkpoint_file_name = args.savedir + os.sep + 'checkpoint_{}.pth.tar'.format(epoch)
        poly_lr_scheduler(args, optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        # train for one epoch
        model.train()
        model.backbone.required_grad = False
        train(args, source_loader,target_loader, model, criteria, optimizer, epoch)
        # model.eval()
        # # validation
        # val(valLoader, model)
        torch.save(model.state_dict(), model_file_name)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr': lr
        }, checkpoint_file_name)


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
    parser.add_argument('--pretrained', default='', help='Pretrained ESPNetv2 weights.')

    train_net(parser.parse_args())
