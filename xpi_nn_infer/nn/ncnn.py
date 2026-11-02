#!/usr/bin/env python3

# https://github.com/Tencent/ncnn

# pip install pnnx ncnn


import torch
import torch.nn as nn
import pnnx

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.mean((2, 3))
        return self.fc(x)

model = Model().eval()

x = torch.rand(1, 3, 224, 224)
pnnx.export(model, "model.pt", (x,))

import numpy as np
import ncnn

net = ncnn.Net()
net.load_param("model.ncnn.param")
net.load_model("model.ncnn.bin")

x = np.zeros((3, 224, 224), np.float32)
mat = ncnn.Mat(x)

ex = net.create_extractor()
ex.input("in0", mat)

ret, out = ex.extract("out0")
print(np.array(out).shape)
