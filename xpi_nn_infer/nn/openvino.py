#!/usr/bin/env python3

from typing import Callable

import numpy as np
from numpy import ndarray
from openvino import Core, Model, CompiledModel

from xpi_nn_infer.nn import Inferer

Callback = Callable[[ndarray, ndarray, int], None]


class OpenvinoModel(Inferer):

  def __init__(self, model_path:str):
    core = Core()
    model = core.read_model(model_path)
    if isinstance(model, Model):
      model = core.compile_model(model)
    self.model: CompiledModel = model
    self.model_path = model_path
    self.outputs0 = self.model.outputs[0]

  def infer(self, x:ndarray):
    return self.model(x)[self.outputs0]

  def infer_softmax(self, x:ndarray):
    logits = self.infer(x)
    if isinstance(logits, dict):
      logits = logits['out']
    return np.argmax(logits, axis=1)


def get_torchvision_model(name:str, input_size:list[int], pretrained:bool=False) -> OpenvinoModel:
  from xpi_nn_infer.nn.torch import get_torchvision_model as get_tc_model
  from xpi_nn_infer.utils import MODEL_PATH, encode_input_size

  ox_fp = MODEL_PATH / f'{name}-{encode_input_size(input_size)}.xml'
  if not ox_fp.is_file():
    tc_model = get_tc_model(name, pretrained)
    tc_model.to_openvino(ox_fp, input_size)
  return OpenvinoModel(ox_fp)
