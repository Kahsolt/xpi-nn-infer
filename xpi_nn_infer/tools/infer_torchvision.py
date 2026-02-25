#!/usr/bin/env python3

from argparse import ArgumentParser


def run(args):
  pass


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-K', default='torch', help='inferer backend')
  parser.add_argument('-M', default='resnet50', help='model arch')
  parser.add_argument('-I', help='input image file or folder path, or "random" for dummy input')
  args = parser.parse_args()

  run(args)
