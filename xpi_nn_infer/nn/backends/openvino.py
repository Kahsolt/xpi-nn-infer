#!/usr/bin/env python3

import numpy as np
from numpy import ndarray
import openvino as ov


class OpenvinoModel:

  pass


class OpenvinoInferer:

  def __init__(self, model_path:str):
    core = ov.Core()
    ov_model = core.read_model(model_path)
    self.compiled_model = core.compile_model(ov_model)

  def __call__(self, x:ndarray):
    x = np.resize(x, (224, 224))
    X = np.expand_dims(np.transpose(x, (2, 1, 0)), 0) # [B=1, C=3, H, W]
    logits = self.compiled_model(X)[0]
    pred = np.argmax(logits, axis=-1)
    return pred
