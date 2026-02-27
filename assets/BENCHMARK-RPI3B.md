### Benchmark - rpi3b

dtype: float32 (fp16/bf16 is x8 slower than fp32 on rpi, DO NOT USE!!)
input_size: 3x224x224

----

- Models and pre-trained weights: https://docs.pytorch.org/vision/main/models.html

ℹ Only concern models with `GFLOPS <= 4.0`, otherwise farrr toooo slow :(

⚪ torchvision.classifier

| Model | GFLOPS | Accuracy (top1/top5) | FPS:torch | FPS:onnx | FPS:openivno |
| :-: | :-: | :-: | :-: | :-: | :-: |
| mobilenet_v3_small | 0.06 | 67.668/87.402 | 6.00 | 12.40 | 24.7 |
| mobilenet_v3_large | 0.22 | 75.274/92.566 | 2.60 |     ? |    ? |
| shufflenet_v2_x2_0 | 0.58 | 76.230/93.006 | 2.22 |     ? |    ? |
| mnasnet1_3         | 0.53 | 76.506/93.522 | 1.60 |     ? |    ? |
| resnet18           | 1.81 | 69.758/89.078 | 1.85 |     ? |    ? |
| resnet50           | 4.09 | 80.858/95.434 | 0.56 |     ? |    ? |
| densenet121        | 2.83 | 74.434/91.972 | 0.70 |     ? |    ? |

⚪ torchvision.segmentation

| Model | GFLOPS | MIoU | PixAcc | FPS:torch | FPS:onnx | FPS:openivno |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| lraspp_mobilenet_v3_large | 2.09 | 57.9 | 91.2 | 2.13 | ? | ? |

⚪ torchvision.detection

| Model | GFLOPS | BoxMAP | FPS:torch | FPS:onnx | FPS:openivno |
| :-: | :-: | :-: | :-: | :-: | :-: |
| fasterrcnn_mobilenet_v3_large_320_fpn | 0.72 | 22.8 | 1.19 | x | x |
| ssdlite320_mobilenet_v3_large         | 0.58 | 21.3 | 0.94 | x | x |

⚪ paddle-ocr

⚪ chinese-ocr
