#!/usr/bin/env python3

import os
from pathlib import Path

BASE_PATH = Path(__file__).parent   # base path for pypi package
ROOT_PATH = BASE_PATH.parent        # root path for git project
IS_PYPI = 'site-packages' in str(BASE_PATH)

# decide model folder
if IS_PYPI:
  MODEL_PATH = BASE_PATH / 'MODEL_PATH'
else:
  MODEL_PATH = ROOT_PATH / 'MODEL_PATH'
# override by envvar
if 'MODEL_PATH' in os.environ:
  MODEL_PATH = Path.expanduser(Path(os.environ['MODEL_PATH']))
  assert MODEL_PATH.is_dir(), f'MODEL_PATH detected in envvar, but path {MODEL_PATH} not exists'


class ValueWindow:

  def __init__(self, nlen:int=100):
    self.nlen = nlen
    self.vals: list[float] = []
    self.sum = 0.0

  def append(self, v:float):
    if len(self.vals) >= self.nlen:
      self.sum -= self.vals.pop(0)
    self.vals.append(v)
    self.sum += v

  @property
  def mean(self) -> float:
    if not self.vals: return 0.0
    return self.sum / len(self.vals)


def encode_input_size(input_size:tuple[int]) -> str:
  '''encode size tuple to safe filename'''
  return '[' + ','.join([str(e) for e in input_size]) + ']'
