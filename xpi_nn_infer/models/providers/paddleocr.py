#!/usr/bin/env python3

from time import time
import numpy as np
import onnxruntime as rt

det = rt.InferenceSession('models/ch_PP-OCRv4_det_infer.onnx')
ts_start = time()
x = np.random.uniform(size=[1, 3, 320, 320]).astype(np.float32)
out = det.run(["sigmoid_0.tmp_0"], {"x": x})
ts_end = time()
print('det:', x.shape, '->', out[0].shape, ', time cost:', ts_end - ts_start)

opts = rt.SessionOptions()
opts.log_severity_level = 3
rec = rt.InferenceSession('models/ch_PP-OCRv4_rec_infer.onnx', opts)
ts_start = time()
x = np.random.uniform(size=[1, 3, 48, 256]).astype(np.float32)
out = rec.run(["softmax_11.tmp_0"], {"x": x})
ts_end = time()
print('rec:', x.shape, '->', out[0].shape, ', time cost:', ts_end - ts_start)
