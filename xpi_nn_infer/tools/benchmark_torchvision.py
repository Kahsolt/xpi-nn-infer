#!/usr/bin/env python3

from argparse import ArgumentParser

import numpy as np


def run_torch(args):
  import torch
  from xpi_nn_infer.nn.torch import get_torchvision_model
  torch.set_num_threads(torch.get_num_interop_threads())
  tc_model = get_torchvision_model(args.model, pretrained=False)
  tc_model.benchmark(torch.rand(args.input_size, dtype=torch.float32), n_iter=args.n_iter)


def run_onnx(args):
  from xpi_nn_infer.nn.onnx import get_torchvision_model
  ox_model = get_torchvision_model(args.model, args.input_size, pretrained=False)
  ox_model.benchmark(np.random.uniform(size=args.input_size).astype(np.float32), n_iter=args.n_iter)


def run_openvino(args):
  from xpi_nn_infer.nn.openvino import get_torchvision_model
  ov_model = get_torchvision_model(args.model, args.input_size, pretrained=False)
  ov_model.benchmark(np.random.uniform(size=args.input_size).astype(np.float32), n_iter=args.n_iter)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-K', '--backend', default='torch', choices=['torch', 'onnx', 'openvino'], help='nn backend')
  parser.add_argument('-M', '--model', default='mobilenet_v3_small', help='model arch')
  parser.add_argument('-I', '--input_size', type=int, nargs='+', default=[1, 3, 224, 224], help='input size')
  parser.add_argument('-N', '--n_iter', type=int, default=3000, help='n_iter')
  args = parser.parse_args()

  print(args)

  runner = globals().get(f'run_{args.backend}')
  try:
    runner(args)
  except KeyboardInterrupt:
    print('>> Exit by Ctrl+C')
