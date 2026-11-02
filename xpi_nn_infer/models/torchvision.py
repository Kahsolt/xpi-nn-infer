#!/usr/bin/env python3

from dataclasses import dataclass

import torch.nn as nn
from torchvision.models import Weights, WeightsEnum
from torchvision.models import list_models, get_model_builder, get_model, get_model_weights
from torchvision.models import MobileNet_V3_Small_Weights, MobileNetV3


@dataclass
class ModelInfo:
  acc_top1: float
  acc_top5: float
  params: float
  gflops: float


# https://docs.pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
MODEL_INFOS = {
  'alexnet': ModelInfo(56.522, 79.066, 61.1, 0.71),
  'convnext_base': ModelInfo(84.062, 96.87, 88.6, 15.36),
  'convnext_large': ModelInfo(84.414, 96.976, 197.8, 34.36),
  'convnext_small': ModelInfo(83.616, 96.65, 50.2, 8.68),
  'convnext_tiny': ModelInfo(82.52, 96.146, 28.6, 4.46),
  'densenet121': ModelInfo(74.434, 91.972, 8.0, 2.83),
  'densenet161': ModelInfo(77.138, 93.56, 28.7, 7.73),
  'densenet169': ModelInfo(75.6, 92.806, 14.1, 3.36),
  'densenet201': ModelInfo(76.896, 93.37, 20.0, 4.29),
  'efficientnet_b0': ModelInfo(77.692, 93.532, 5.3, 0.39),
  'efficientnet_b1': ModelInfo(78.642, 94.186, 7.8, 0.69),
  'efficientnet_b2': ModelInfo(80.608, 95.31, 9.1, 1.09),
  'efficientnet_b3': ModelInfo(82.008, 96.054, 12.2, 1.83),
  'efficientnet_b4': ModelInfo(83.384, 96.594, 19.3, 4.39),
  'efficientnet_b5': ModelInfo(83.444, 96.628, 30.4, 10.27),
  'efficientnet_b6': ModelInfo(84.008, 96.916, 43.0, 19.07),
  'efficientnet_b7': ModelInfo(84.122, 96.908, 66.3, 37.75),
  'efficientnet_v2_l': ModelInfo(85.808, 97.788, 118.5, 56.08),
  'efficientnet_v2_m': ModelInfo(85.112, 97.156, 54.1, 24.58),
  'efficientnet_v2_s': ModelInfo(84.228, 96.878, 21.5, 8.37),
  'googlenet': ModelInfo(69.778, 89.53, 6.6, 1.5),
  'inception_v3': ModelInfo(77.294, 93.45, 27.2, 5.71),
  'mnasnet0_5': ModelInfo(67.734, 87.49, 2.2, 0.1),
  'mnasnet0_75': ModelInfo(71.18, 90.496, 3.2, 0.21),
  'mnasnet1_0': ModelInfo(73.456, 91.51, 4.4, 0.31),
  'mnasnet1_3': ModelInfo(76.506, 93.522, 6.3, 0.53),
  'maxvit_t': ModelInfo(83.7, 96.722, 30.9, 5.56),
  'mobilenet_v2': ModelInfo(71.878, 90.286, 3.5, 0.3),
  'mobilenet_v3_large': ModelInfo(74.042, 91.34, 5.5, 0.22),
  'mobilenet_v3_small': ModelInfo(67.668, 87.402, 2.5, 0.06),
  'regnet_x_16gf': ModelInfo(80.058, 94.944, 54.3, 15.94),
  'regnet_x_1_6gf': ModelInfo(77.04, 93.44, 9.2, 1.6),
  'regnet_x_32gf': ModelInfo(80.622, 95.248, 107.8, 31.74),
  'regnet_x_3_2gf': ModelInfo(78.364, 93.992, 15.3, 3.18),
  'regnet_x_400mf': ModelInfo(72.834, 90.95, 5.5, 0.41),
  'regnet_x_800mf': ModelInfo(75.212, 92.348, 7.3, 0.8),
  'regnet_x_8gf': ModelInfo(79.344, 94.686, 39.6, 8),
  'regnet_y_128gf': ModelInfo(86.068, 97.844, 644.8, 127.52),
  'regnet_y_16gf': ModelInfo(80.424, 95.24, 83.6, 15.91),
  'regnet_y_1_6gf': ModelInfo(77.95, 93.966, 11.2, 1.61),
  'regnet_y_32gf': ModelInfo(80.878, 95.34, 145.0, 32.28),
  'regnet_y_3_2gf': ModelInfo(78.948, 94.576, 19.4, 3.18),
  'regnet_y_400mf': ModelInfo(74.046, 91.716, 4.3, 0.4),
  'regnet_y_800mf': ModelInfo(76.42, 93.136, 6.4, 0.83),
  'regnet_y_8gf': ModelInfo(80.032, 95.048, 39.4, 8.47),
  'resnext101_32x8d': ModelInfo(79.312, 94.526, 88.8, 16.41),
  'resnext101_64x4d': ModelInfo(83.246, 96.454, 83.5, 15.46),
  'resnext50_32x4d': ModelInfo(77.618, 93.698, 25.0, 4.23),
  'resnet101': ModelInfo(77.374, 93.546, 44.5, 7.8),
  'resnet152': ModelInfo(78.312, 94.046, 60.2, 11.51),
  'resnet18': ModelInfo(69.758, 89.078, 11.7, 1.81),
  'resnet34': ModelInfo(73.314, 91.42, 21.8, 3.66),
  'resnet50': ModelInfo(76.13, 92.862, 25.6, 4.09),
  'shufflenet_v2_x0_5': ModelInfo(60.552, 81.746, 1.4, 0.04),
  'shufflenet_v2_x1_0': ModelInfo(69.362, 88.316, 2.3, 0.14),
  'shufflenet_v2_x1_5': ModelInfo(72.996, 91.086, 3.5, 0.3),
  'shufflenet_v2_x2_0': ModelInfo(76.23, 93.006, 7.4, 0.58),
  'squeezenet1_0': ModelInfo(58.092, 80.42, 1.2, 0.82),
  'squeezenet1_1': ModelInfo(58.178, 80.624, 1.2, 0.35),
  'swin_b': ModelInfo(83.582, 96.64, 87.8, 15.43),
  'swin_s': ModelInfo(83.196, 96.36, 49.6, 8.74),
  'swin_t': ModelInfo(81.474, 95.776, 28.3, 4.49),
  'swin_v2_b': ModelInfo(84.112, 96.864, 87.9, 20.32),
  'swin_v2_s': ModelInfo(83.712, 96.816, 49.7, 11.55),
  'swin_v2_t': ModelInfo(82.072, 96.132, 28.4, 5.94),
  'vgg11_bn': ModelInfo(70.37, 89.81, 132.9, 7.61),
  'vgg11': ModelInfo(69.02, 88.628, 132.9, 7.61),
  'vgg13_bn': ModelInfo(71.586, 90.374, 133.1, 11.31),
  'vgg13': ModelInfo(69.928, 89.246, 133.0, 11.31),
  'vgg16_bn': ModelInfo(73.36, 91.516, 138.4, 15.47),
  'vgg16': ModelInfo(71.592, 90.382, 138.4, 15.47),
  'vgg19_bn': ModelInfo(74.218, 91.842, 143.7, 19.63),
  'vgg19': ModelInfo(72.376, 90.876, 143.7, 19.63),
  'vit_b_16': ModelInfo(81.072, 95.318, 86.6, 17.56),
  'vit_b_32': ModelInfo(75.912, 92.466, 88.2, 4.41),
  'vit_l_16': ModelInfo(79.662, 94.638, 304.3, 61.55),
  'vit_l_32': ModelInfo(76.972, 93.07, 306.5, 15.38),
  'wide_resnet101_2': ModelInfo(78.848, 94.284, 126.9, 22.75),
  'wide_resnet50_2': ModelInfo(78.468, 94.086, 68.9, 11.4),
}


def list_model_weights(name:str) -> list[WeightsEnum]:
  return list(get_model_weights(name))


def get_pretrained_model(name:str, weights:WeightsEnum='auto') -> nn.Module:
  if weights == 'auto':
    weights = get_model_weights(name)['DEFAULT']
  return get_model(name, weights=weights)


def get_mbv3_small(num_classes:int=10) -> MobileNetV3:
  model: MobileNetV3 = get_pretrained_model('mobilenet_v3_small', MobileNet_V3_Small_Weights.IMAGENET1K_V1)
  linear_old: nn.Linear = model.classifier[-1]
  linear_new = nn.Linear(linear_old.in_features, num_classes, bias=linear_old.bias is not None)
  model.classifier[-1] = linear_new
  return model
