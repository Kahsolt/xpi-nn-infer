#!/usr/bin/env python3

from argparse import ArgumentParser

from time import sleep
import numpy as np

from xpi_nn_infer.torch_utils import DEVICES, DTYPES, torch_backends_info

MODELS = [
  'mobilenet_v3_small',
  'mobilenet_v3_large',
  'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0',
  'squeezenet1_1',
  'efficientnet_v2_s',
  'vgg16_bn',
  'resnet18',
  'resnet50',
  'densenet121',
  'resnext50_32x4d',
  'convnext_tiny',
  'vit_b_16',
  'vit_b_32',
  'swin_t',
  'swin_v2_t',
]


def run_torch(args):
  import torch
  from xpi_nn_infer.nn.torch import get_torchvision_model
  from xpi_nn_infer.torch_utils import resolve_device, resolve_dtype
  device = resolve_device(args.device)
  dtype = resolve_dtype(args.dtype)
  tc_model = get_torchvision_model(args.model, pretrained=False).to(dtype, device)
  fps = tc_model.benchmark(torch.rand(args.input_size, dtype=dtype, device=device), n_iter=args.n_iter)
  return fps


def run_onnx(args):
  from xpi_nn_infer.nn.onnx import get_torchvision_model
  ox_model = get_torchvision_model(args.model, args.input_size, pretrained=False)
  fps = ox_model.benchmark(np.random.uniform(size=args.input_size).astype(np.float32), n_iter=args.n_iter)
  return fps


def run_openvino(args):
  from xpi_nn_infer.nn.openvino import get_torchvision_model
  ov_model = get_torchvision_model(args.model, args.input_size, pretrained=False)
  fps = ov_model.benchmark(np.random.uniform(size=args.input_size).astype(np.float32), n_iter=args.n_iter)
  return fps


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-K', '--backend', default='torch', choices=['torch', 'onnx', 'openvino'], help='nn backend')
  parser.add_argument('-M', '--model', default='mobilenet_v3_small', choices=MODELS, help='model arch')
  parser.add_argument('-D', '--device', default='auto', choices=DEVICES, help='device')
  parser.add_argument('-T', '--dtype', default='fp32', choices=DTYPES.keys(), help='dtype')
  parser.add_argument('-I', '--input_size', type=int, nargs='+', default=[1, 3, 224, 224], help='input size')
  parser.add_argument('-N', '--n_iter', type=int, default=1500, help='n_iter')
  parser.add_argument('-X', '--run_all', action='store_true')
  args = parser.parse_args()

  torch_backends_info()
  print(args)

  runner = globals().get(f'run_{args.backend}')
  if args.run_all:
    results = []
    for model in MODELS:
      print(f'>> run model: {model}')
      args.model = model
      fps = runner(args)
      results.append((model, fps))
      sleep(10)
    print()
    print('=' * 72)
    print(f'| Model | FPS |')
    print(f'| :-: | :-: |')
    for model, fps in results:
      print(f'| {model} | {fps:.2f} |')
  else:
    try:
      runner(args)
    except KeyboardInterrupt:
      print('>> Exit by Ctrl+C')
