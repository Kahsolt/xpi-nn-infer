#!/usr/bin/env python3

import os
from pathlib import Path

BASE_PATH = Path(__file__).parent

if 'MODEL_PATH' in os.environ:
  MODEL_PATH = Path.expanduser(os.environ['MODEL_PATH'])
  assert MODEL_PATH.is_dir(), f'MODEL_PATH detected in envvar, but path {MODEL_PATH!r} not exists'
else:
  MODEL_PATH = BASE_PATH / 'MODEL_PATH'
  MODEL_PATH.mkdir(exist_ok=True)


class ValueWindow:

  def __init__(self, nlen:int=100):
    self.nlen = nlen
    self.vals = []
    self.sum = 0.0

  def append(self, v:float):
    if len(self.vals) >= self.nlen:
      self.sum -= self.vals.pop(0)
    else:
      self.vals.append(v)
    self.sum += v

  @property
  def mean(self) -> float:
    if not self.vals: return 0.0
    return self.sum / len(self.vals)
