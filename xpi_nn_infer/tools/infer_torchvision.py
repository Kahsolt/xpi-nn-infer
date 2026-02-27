#!/usr/bin/env python3

from pathlib import Path
from time import time
from argparse import ArgumentParser
from typing import Generator

import numpy as np
from numpy import ndarray

from xpi_nn_infer.io.PIL import imread

# [YieldType, SendType, ReturnType]
DataGen = Generator[ndarray, None, None]


def prepare_input(args) -> DataGen:
  if args.input == "random":
    print(f'>> WARN: using random input size {args.input_size}')
    yield np.random.uniform(size=args.input_size).astype(np.float32)
  else:
    resize = args.input_size[-2:]
    path = Path(args.input)
    if path.is_file():
      yield imread(args.input, resize=resize, mode='RGB', batchify=True)
    elif path.is_dir():
      for fp in path.iterdir():
        if not fp.is_file(): continue
        try:
          yield imread(fp, resize=resize, mode='RGB', batchify=True)
        except Exception as e:
          print(f'skip path {fp}, error {e}')
    else:
      raise ValueError(f'>> --input is neither file nor folder: {args.input}')


def run_torch(args, x:DataGen):
  import torch
  from xpi_nn_infer.nn.torch import get_torchvision_model
  torch.set_num_threads(torch.get_num_interop_threads())
  tc_model = get_torchvision_model(args.model, pretrained=True)
  for x in xgen:
    pred = tc_model.infer_softmax(torch.from_numpy(x))
    print('pred:', pred[0].item())


def run_onnx(args, x:DataGen):
  from xpi_nn_infer.nn.onnx import get_torchvision_model
  ox_model = get_torchvision_model(args.model, args.input_size, pretrained=True)
  for x in xgen:
    pred = ox_model.infer_softmax(x)
    print('pred:', pred[0].item())


def run_openvino(args, x:DataGen):
  from xpi_nn_infer.nn.openvino import get_torchvision_model
  ov_model = get_torchvision_model(args.model, args.input_size, pretrained=True)
  for x in xgen:
    pred = ov_model.infer_softmax(x)
    print('pred:', pred[0].item())


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-K', '--backend', default='torch', choices=['torch', 'onnx', 'openvino'], help='nn backend')
  parser.add_argument('-M', '--model', default='mobilenet_v3_small', help='model arch')
  parser.add_argument('-I', '--input', default='random', help='input image file or folder path, or "random" for dummy input')
  parser.add_argument('-S', '--input_size', type=int, nargs='+', default=[1, 3, 224, 224], help='input size')
  args = parser.parse_args()

  runner = globals().get(f'run_{args.backend}')

  ts_start = time()
  xgen = prepare_input(args)
  runner(args, xgen)
  ts_end = time()
  print(f'>> Timecost: {ts_end - ts_start:.3f}s')
