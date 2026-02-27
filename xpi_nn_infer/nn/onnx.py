#!/usr/bin/env python3

from pathlib import Path
from typing import Callable

import numpy as np
from numpy import ndarray
from onnxruntime import InferenceSession, SessionOptions

from xpi_nn_infer.nn import Inferer


Callback = Callable[[ndarray, ndarray, int], None]


class OnnxModel(Inferer):

  def __init__(self, model_path:str, options:SessionOptions=None):
    options = options or SessionOptions()
    options.log_severity_level = 3

    self.model: InferenceSession = InferenceSession(model_path, options)
    self.model_path = model_path
    self.input_name0 = self.model.get_inputs()[0].name
    self.output_name0 = self.model.get_outputs()[0].name

  def infer(self, x:ndarray) -> ndarray:
    return self.model.run([self.output_name0], {self.input_name0: x})[0]

  def infer_softmax(self, x:ndarray):
    logits = self.infer(x)
    if isinstance(logits, dict):
      logits = logits['out']
    return np.argmax(logits, axis=1)

  def to_openvino(self, fp:str):
    import openvino as ov

    assert Path(fp).suffix == '.xml', 'fp suffix should be .xml'
    ov_model = ov.convert_model(self.model)
    ov.save_model(ov_model, fp)


def get_torchvision_model(name:str, input_size:list[int], pretrained:bool=False) -> OnnxModel:
  from xpi_nn_infer.nn.torch import get_torchvision_model as get_tc_model
  from xpi_nn_infer.utils import MODEL_PATH, encode_input_size

  ox_fp = MODEL_PATH / f'{name}-{encode_input_size(input_size)}.onnx'
  if not ox_fp.is_file():
    tc_model = get_tc_model(name, pretrained)
    tc_model.to_onnx(ox_fp, input_size)
  return OnnxModel(ox_fp)
