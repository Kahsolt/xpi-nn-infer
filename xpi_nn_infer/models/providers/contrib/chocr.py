#!/usr/bin/env python3

from time import time
import numpy as np
import onnxruntime as rt

det = rt.InferenceSession('models/dbnet.onnx')
ts_start = time()
x = np.random.uniform(size=[1, 3, 320, 320]).astype(np.float32)
out = det.run(["out1"], {"input0": x})
ts_end = time()
print('det:', x.shape, '->', out[0].shape, ', time cost:', ts_end - ts_start)

opts = rt.SessionOptions()
opts.log_severity_level = 3
rec = rt.InferenceSession('models/crnn_lite_lstm.onnx', opts)
ts_start = time()
x = np.random.uniform(size=[1, 3, 32, 256]).astype(np.float32)
out = rec.run(["out"], {"input": x})
ts_end = time()
print('rec:', x.shape, '->', out[0].shape, ', time cost:', ts_end - ts_start)

cls = rt.InferenceSession('models/angle_net.onnx')
ts_start = time()
x = np.random.uniform(size=[1, 3, 32, 192]).astype(np.float32)
out = cls.run(["out"], {"input": x})
ts_end = time()
print('cls:', x.shape, '->', out[0].shape, ', time cost:', ts_end - ts_start)
