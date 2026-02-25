#!/usr/bin/env python3

#!/usr/bin/env python3
# Author: Armit
# Create Time: 周五 2025/03/14 

from pathlib import Path
from time import time
from argparse import ArgumentParser

# 跑 torchvision 的预训练模型

BASE_PATH = Path(__file__).parent
MODEL_PATH = BASE_PATH / 'models'

MODELS = [
  'mobilenet_v3_small',
  'mobilenet_v3_large',
  'shufflenet_v2_x2_0',
  'resnet18',
  'resnet50',
  'densenet121',
  'lraspp_mobilenet_v3_large',
  'fasterrcnn_mobilenet_v3_large_320_fpn',
  'fasterrcnn_mobilenet_v3_large_fpn',
  'ssdlite320_mobilenet_v3_large',
]
BACKENDS = [
  'torch',
  'onnx',
  'openvino',
]

def run_torch(args):
  '''
  NOTE: fp16/bf16 is x8 slower than fp32 on rpi, DO NOT USE!!

  input size: 3x224x224

  | model | Accuracy (top1/top5) | GFLOPS | FPS |
  | :-: | :-: | :-: | :-: |
  | mobilenet_v3_small | 67.668/87.402 | 0.06 | 6.00 |
  | mobilenet_v3_large | 75.274/92.566 | 0.22 | 2.60 |
  | shufflenet_v2_x2_0 | 76.230/93.006 | 0.58 | 2.22 |
  | mnasnet1_3         | 76.506/93.522 | 0.53 | 1.60 |
  | resnet18           | 69.758/89.078 | 1.81 | 1.85 |
  | resnet50           | 80.858/95.434 | 4.09 | 0.56 |
  | densenet121        | 74.434/91.972 | 2.83 | 0.70 |

  | model | MIoU | PixAcc | GFLOPS | FPS |
  | :-: | :-: | :-: | :-: | :-: |
  | lraspp_mobilenet_v3_large | 57.9 | 91.2 | 2.09 | 2.13 |

  | model | BoxMAP | GFLOPS | FPS |
  | :-: | :-: | :-: | :-: |
  | fasterrcnn_mobilenet_v3_large_320_fpn | 22.8 | 0.72 | 1.19 |
  | fasterrcnn_mobilenet_v3_large_fpn     | 32.8 | 4.49 | 0.20 |
  | ssdlite320_mobilenet_v3_large         | 21.3 | 0.58 | 0.94 |
  '''

  import torch
  import torchvision.models as M

  #device = 'cpu'
  device = 'cuda'
  if device == 'cuda':
    dtype = torch.float32
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
  else:
    dtype = torch.float32

  print('Load model')
  #model = M.mobilenet_v3_small(pretrained=False)
  #model = M.mobilenet_v3_large(pretrained=False)
  #model = M.shufflenet_v2_x2_0(pretrained=True)
  #model = M.mnasnet1_3(pretrained=True)
  #model = M.resnet18(pretrained=True)
  #model = M.resnet50(pretrained=True)
  #model = M.densenet121(pretrained=True)
  #model = M.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
  #model = M.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
  #model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
  #model = M.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
  model = M.vgg16(pretrained=False)
  model = model.to(dtype=dtype, device=device).eval()

  @torch.inference_mode()
  def run():
    wv = ValueWindow()
    i = 0
    while True:
      ts_start = time()

      X = torch.rand([1, 3, 224, 224], dtype=dtype, device=device)
      logits = model(X)

      if isinstance(logits, torch.Tensor):
        pred = logits.argmax(-1)
      elif isinstance(logits, dict):
        logits = logits['out']
        pred = logits.argmax(-1)

      if device == 'cuda':
        torch.cuda.synchronize()

      ts_cost = time() - ts_start
      wv.append(ts_cost)
      i += 1
      if i % 10 == 0:
        print(f'>> infer time: {wv.avg:.3f}s, FPS: {1 / wv.avg:.2f}')

  run()


def run_onnx(args):
  import numpy as np
  import onnxruntime as rt

  '''
  NOTE: fp16/bf16 is slower than fp32 on rpi, DO NOT USE!!

  input size: 3x224x224

  | model | Accuracy (top1/top5) | GFLOPS | FPS |
  | :-: | :-: | :-: | :-: |
  | mobilenet_v3_small | 67.668/87.402 | 0.06 | 12.40 |
  | mobilenet_v3_large | 75.274/92.566 | 0.22 |  |
  | shufflenet_v2_x2_0 | 76.230/93.006 | 0.58 |  |
  | mnasnet1_3         | 76.506/93.522 | 0.53 |  |
  | resnet18           | 69.758/89.078 | 1.81 |  |
  | resnet50           | 80.858/95.434 | 4.09 |  |
  | densenet121        | 74.434/91.972 | 2.83 |  |

  | model | MIoU | PixAcc | GFLOPS | FPS |
  | :-: | :-: | :-: | :-: | :-: |
  | lraspp_mobilenet_v3_large | 57.9 | 91.2 | 2.09 |  |

  | model | BoxMAP | GFLOPS | FPS |
  | :-: | :-: | :-: | :-: |
  | fasterrcnn_mobilenet_v3_large_320_fpn | 22.8 | 0.72 |  |
  | fasterrcnn_mobilenet_v3_large_fpn     | 32.8 | 4.49 |  |
  | ssdlite320_mobilenet_v3_large         | 21.3 | 0.58 |  |
  '''

  opts = rt.SessionOptions()
  opts.log_severity_level = 3
  model = rt.InferenceSession(args.model_path / f'{args.model}.onnx', opts)
  input_name = model.get_inputs()[0].name

  while True:
    ts_start = time()

    X = np.random.uniform(size=[1, 3, 224, 224]).astype(np.float32)
    logits = model.run(None, {input_name: X})[0]
    pred = np.argmax(logits, axis=-1)

    ts_cost = time() - ts_start
    print(f'>> infer time: {ts_cost:.3f}s, FPS: {1 / ts_cost:.2f}')


def run_openvino(args):
  from time import time
  import numpy as np
  import openvino as ov

  # FPS: 24.7

  core = ov.Core()
  ov_model = core.read_model(args.model_path / f'{args.model}.xml')
  compiled_model = core.compile_model(ov_model)

  while True:
    ts_start = time()

    X = np.random.uniform(size=[1, 3, 224, 224]).astype(np.float32)
    logits = compiled_model(X)
    pred = np.argmax(logits, axis=-1)

    ts_cost = time() - ts_start
    print(f'>> infer time: {ts_cost:.3f}s, FPS: {1 / ts_cost:.2f}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model',   default='mobilenet_v3_small', choices=MODELS)
  parser.add_argument('-K', '--backend', default='openvino',           choices=BACKENDS)
  parser.add_argument('--model_path',    default=MODEL_PATH, type=Path)
  args = parser.parse_args()

  try:
    print('Start run infer!')
    runner = globals()[f'run_{args.backend}']
    runner(args)
  except KeyboardInterrupt:
    pass
