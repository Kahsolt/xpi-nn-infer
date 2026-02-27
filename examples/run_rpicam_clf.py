#!/usr/bin/env python3

# 树莓派3B实时图像分类: 摄像头 -> 分类模型

from time import time
from importlib import import_module
from argparse import ArgumentParser

import numpy as np

from xpi_nn_infer.io.rpicam import get_camera
from xpi_nn_infer.models.torchvision import list_models
from xpi_nn_infer.nn import Inferer
from xpi_nn_infer.utils import ValueWindow


def run(args):
  nn_backend = import_module('.' + args.backend, 'xpi_nn_infer.nn')
  model: Inferer = nn_backend.get_torchvision_model(args.model)

  cam = get_camera(args.mode, args.fps, args.buffers)
  try:
    cam.start()
    vw = ValueWindow()
    while True:
      ts_start = time()
      x = cam.capture_array()         # [W, H, C=3]
      x = np.transpose(x, (2, 0, 1))  # [C=3, W, H]
      x = np.expand_dims(x, 0)        # [B, C=3, W, H]
      pred = model.infer(x)
      ts_end = time()
      vw.append(ts_end - ts_start)

      print(f'>> pred: {pred}, fps: {1 / (vw.mean):.2f}')
  except KeyboardInterrupt:
    print('>> Exit by Ctrl+C')
  finally:
    cam.stop()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-K', '--backend', default='openvino', choices=['torch', 'onnx', 'openvino'])
  parser.add_argument('-M', '--model', default='mobilenet_v3_small', choices=list_models())
  parser.add_argument('-m', '--mode', type=int, default=0, choices=[0, 1, 2, 3], help='camera sensor mode')
  parser.add_argument('-F', '--fps', type=int, default=30, help='fps limit')
  parser.add_argument('-B', '--buffers', type=int, default=5, help='buffer count')
  args = parser.parse_args()

  run(args)
