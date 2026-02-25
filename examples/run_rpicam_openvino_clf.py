#!/usr/bin/env python3

# 树莓派实时图像分类: 摄像头 -> 模型推理 (OpenVino)
# default setting: fps ~= 20

from time import time
from pathlib import Path
from argparse import ArgumentParser

from xpi_nn_infer.io.backends.rpicam import get_camera
from xpi_nn_infer.nn.backends.openvino import OpenvinoInferer
from xpi_nn_infer.utils import ValueWindow, MODEL_PATH


def run(args):
  cam = get_camera(args.mode, args.fps, args.buffers)
  inferer = OpenvinoInferer(args.model_path / f'{args.model}.xml')
  vw = ValueWindow()

  try:
    while True:
      ts_start = time()
      x = cam.capture_array() # [W, H, C=3]
      pred = inferer(x)
      ts_end = time()
      vw.append(ts_end - ts_start)
      print(f'>> pred: {pred}, fps: {1 / (vw.mean):.2f}')
  except KeyboardInterrupt:
    print('>> Exit by Ctrl+C')
  finally:
    cam.stop()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='mobilenet_v3_small')
  parser.add_argument('-D', '--model_path', type=Path, default=MODEL_PATH)
  parser.add_argument('-m', '--mode', type=int, default=0, choices=[0, 1, 2, 3], help='sensor mode')
  parser.add_argument('-F', '--fps', type=int, default=30, help='fps limit')
  parser.add_argument('-B', '--buffers', type=int, default=5, help='buffer count')
  args = parser.parse_args()

  run(args)
