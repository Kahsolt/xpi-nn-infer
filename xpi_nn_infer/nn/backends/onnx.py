#!/usr/bin/env python3


class OnnxModel:

  def __init__(self):
    pass


class OnnxInferer:

  def infer(self, x):
    pass

  def infer_batch(self, dataset, batch_size:int=16):
    pass

  def benchmark(self, datasrc):
    pass


def to_openvino(fp_in:str, fp_out:str):
  import numpy as np
  import openvino as ov

  ov_model = ov.convert_model(fp_in)
  ov.save_model(ov_model, fp_out)

  compiled_model = ov.compile_model(ov_model)
  input_data = np.random.rand(1, 3, 224, 224)
  result = compiled_model(input_data)
