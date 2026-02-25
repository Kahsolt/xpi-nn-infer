#!/usr/bin/env python3

from argparse import ArgumentParser

from xpi_nn_infer.utils import ValueWindow


def run(args):
  vw = ValueWindow()
  pass


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-K', default='torch', help='inferer backend')
  parser.add_argument('-M', default='resnet50', help='model arch')
  args = parser.parse_args()

  run(args)
