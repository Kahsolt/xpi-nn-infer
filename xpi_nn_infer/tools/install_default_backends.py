#!/usr/bin/env python3

import os
from argparse import ArgumentParser

DEFAULT_PACKAGES = [
  'torch',
  'torchvision',
  'onnx',
  'onnxruntime',
  'openvino',
]


def run(args):
  cmd = f'pip install {" ".join(DEFAULT_PACKAGES)}'
  print(f'>> Run cmd: "{cmd}"')
  if not args.quiet:
    s = input('>> Confirm [y/N]? ').strip()
    if s not in ['y', 'Y']:
      print('>> Cancelled.')
      return

  ret = os.system(cmd)
  if ret == 0:
    print('>> Done! :)')
  else:
    print(f'>> [Error] retcode={ret}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-y', '--quiet', action='store_true', help='skip confirm')
  args, _ = parser.parse_known_args()

  run(args)
