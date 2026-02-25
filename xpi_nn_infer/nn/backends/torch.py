#!/usr/bin/env python3

import torch
import torch.nn as nn


class TorchModel:

  def __init__(self, model: nn.Module):
    self.model = model


class TorchInferer:

  def __init__(self, model: TorchModel):
    self.model = model


class TorchConverter:

  def __init__(self, model: TorchModel):
    self.model = model

  def to_onnx(self, fp:str, input_size:tuple[int]=(1, 3, 224, 224)):
    import torch.onnx

    dummy_input = torch.zeros(*input_size)
    torch.onnx.export(
      self.model.model,
      dummy_input,
      fp,
      export_params=True,
      opset_version=13,
    )

  def to_openvino(self, fp:str, input_size:tuple[int]=(1, 3, 224, 224)):
    import openvino as ov

    dummy_input = torch.zeros(*input_size)
    ov_model = ov.convert_model(
      self.model.model,
      example_input=dummy_input,
    )
    ov.save_model(ov_model, fp)

    # TODO：可以save compiled_model吗?
    compiled_model = ov.compile_model(ov_model)
    result = compiled_model(dummy_input)
