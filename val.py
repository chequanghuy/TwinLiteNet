import torch
import torch
from model import TwinLite as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import val, netParams
import torch.optim.lr_scheduler
from const import *


def validation(args):
    '''
    Main function for trainign and validation
    :param args: global arguments
    :return: None
    '''

    # load the model
    model = net.TwinLiteNet()
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True
        

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(valid=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    example = torch.rand(1, 3, 360, 640).cuda()
    model = torch.jit.trace(model, example)
    val(valLoader, model)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--weight', default="pretrained/best.pth")
    parser.add_argument('--num_workers', type=int, default=12, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                                   'Change as per the GPU memory')

    validation(parser.parse_args())
