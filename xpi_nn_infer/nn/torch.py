#!/usr/bin/env python3

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from xpi_nn_infer.nn import Inferer

Callback = Callable[[Tensor, Tensor, int], None]


class TorchModel(Inferer):

  def __init__(self, model:nn.Module):
    self.model = model.eval()

  @torch.inference_mode()
  def infer(self, x:Tensor) -> Tensor:
    return self.model(x)

  def infer_softmax(self, x:Tensor) -> Tensor:
    logits = self.infer(x)
    if isinstance(logits, dict):
      logits = logits['out']
    return torch.argmax(logits, dim=1)

  def to_jit(self, fp:str, input_size:tuple[int]):
    assert Path(fp).suffix == '.pt', 'fp suffix should be .pt'
    traced_model = torch.jit.trace(self.model.eval(), example_inputs=input_size)
    torch.jit.save(traced_model, fp)

  def to_onnx(self, fp:str, input_size:tuple[int], op_ver:int=13):
    import torch.onnx

    assert Path(fp).suffix == '.onnx', 'fp suffix should be .onnx'
    dummy_input = torch.zeros(*input_size)
    torch.onnx.export(self.model, dummy_input, fp, opset_version=op_ver, export_params=True, dynamo=False)

  def to_openvino(self, fp:str, input_size:tuple[int]):
    import openvino as ov

    assert Path(fp).suffix == '.xml', 'fp suffix should be .xml'
    dummy_input = torch.zeros(*input_size)
    ov_model = ov.convert_model(self.model, example_input=dummy_input)
    ov.save_model(ov_model, fp)


def get_torchvision_model(name:str, pretrained:bool=False) -> TorchModel:
  from xpi_nn_infer.models.torchvision import get_model, get_pretrained_model

  if pretrained:
    model = get_pretrained_model(name)
  else:
    model = get_model(name)
  return TorchModel(model)
