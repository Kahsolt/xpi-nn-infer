#!/usr/bin/env python3

# https://github.com/alibaba/MNN
# https://mnn-docs.readthedocs.io/en/latest/

# pip install -U MNN

import MNN
import MNN.expr as nn
import numpy as np

# 1. Load the pre-compiled MNN model graph
# The Module interface manages runtime memory and execution backend automatically
net = nn.load_module_from_file("model.mnn", ["input_tensor_name"], ["output_tensor_name"])

# 2. Prepare the input data using NumPy
raw_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 3. Convert the NumPy array into an MNN Var object
input_var = nn.placeholder([1, 3, 224, 224], nn.NCHW)
input_var.write(raw_data.tobytes())

# 4. Run inference 
outputs = net.forward([input_var])

# 5. Extract output data back into NumPy format
output_var = outputs[0]
output_data = output_var.read()  # Returns raw bytes
output_numpy = np.frombuffer(output_data, dtype=np.float32)
print("Inference Success! Shape:", output_var.shape)
