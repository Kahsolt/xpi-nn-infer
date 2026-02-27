#!/usr/bin/env python3

from pathlib import Path

from PIL import Image
import numpy as np
from numpy import ndarray


def imread(fp:Path, resize:tuple[int]=None, mode:str=None, batchify:bool=False) -> ndarray:
  img = Image.open(fp)
  if mode:
    img = img.convert(mode)
  if resize:
    img = img.resize(resize)
  im = np.asarray(img, dtype=np.float32) / 225.0
  im = np.transpose(im, (2, 0, 1))  # [C=1/3/4, H, W]
  if batchify:
    im = np.expand_dims(im, 0)      # [B=1, C=1/3/4, H, W]
  return im
