import torch
import numpy as np
import argparse
# from tqdm.autonotebook import tqdm
import os
import torch
from model.TwinLite import TwinLiteNet as Net
import onnx
import torch

import pandas as pd
from pathlib import Path



def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = f'TwinLiteNet 🚀 torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    # LOGGER.info(s)
    return torch.device(arg)





def export_formats():
    # YOLOv5 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['TensorRT', 'engine', '.engine', False, True],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

def export_torchscript(model, im, file):
    f = file.with_suffix('.torchscript')
    ts = torch.jit.trace(model, im)
    ts.save(str(f))
def export_onnx(model, im, file, opset=12):
    f = file.with_suffix('.onnx')
    torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["da", "ll"])
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, f)


def export_engine(model, im, file):
    import tensorrt as trt
    export_onnx(model, im, file, 12)  # opset 12
    onnx = file.with_suffix('.onnx')
    f = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    
    # logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(
            f' input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(
            f'output "{out.name}" with shape{out.shape} {out.dtype}')


    print(
        f'building FP {16} engine as {f}')
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())



def run(
        weights='',
        imgsz=[],
        batch_size=1,
        device='',
        include=[]
    ):
    model = net.Net()
    model = model.cuda()
    model.load_state_dict(torch.load(weights))
    device = select_device(device)

    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    jit, onnx, engine = flags  # export booleans

    file = Path(weights)  # PyTorch weights
    # Input
    imgsz = [x for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection
    # Exports
    f = [''] * len(fmts)  # exported filenames
    # warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    if jit:  # TorchScript
        export_torchscript(model, im, file)
    if engine:  # TensorRT required before ONNX
        export_engine(model, im, file)
    if onnx:  # OpenVINO requires ONNX
        export_onnx(model, im, file)




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='pretrained/model_best.pth', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[360, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--include',
        nargs='+',
        default=['engine'],
        help='torchscript, onnx, engine')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

