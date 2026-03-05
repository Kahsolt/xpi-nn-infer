#!/usr/bin/env python3

import os

import torch
from torch import Tensor
from numpy import ndarray

from xpi_nn_infer.utils import exit_missing_packages

device = 'cuda' if torch.cuda.is_available() else 'cpu'

HF_ENDPOINT = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = HF_ENDPOINT

# Mainly focus on tiny/small models (1~4G RAM runnable)
HF_MODELS = {
  'ae': [
    'stabilityai/sdxl-vae',                 # 335 MB
    'stabilityai/sd-vae-ft-mse',            # 335 MB
    'stabilityai/sd-vae-ft-ema',            # 335 MB
    'stabilityai/sd-vae-ft-mse-original',   # 335 MB
    'stabilityai/sd-vae-ft-ema-original',   # 335 MB

    'madebyollin/sdxl-vae-fp16-fix',        # 335 MB
    'madebyollin/taef2',                    # 10.7 MB (need wrapper code!!)
    'madebyollin/taef1',                    # 9.85 MB
    'madebyollin/taesd3',                   # 9.85 MB
    'madebyollin/taesdxl',                  # 10 MB
    'madebyollin/taesd',                    # 10 MB
  ],
}


class TinyVAE:

  def __init__(self, model_path:str='madebyollin/taesd'):
    from diffusers import AutoencoderTiny

    self.model_path = model_path
    self.model = AutoencoderTiny.from_pretrained(model_path)
    self.model.config.shift_factor = 0.0

  def infer(self, x:Tensor) -> Tensor:
    return self.model(x)
