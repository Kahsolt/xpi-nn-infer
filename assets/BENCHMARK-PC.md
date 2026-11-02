### performance benchmark (PC: desktop & mobile)

    仅关心训练速度，用于评估硬件性能 :)

----

ℹ 评测条件: `用torchvision模型从头训练随机数据(224x224/1000分类/bs=32)，取3个epoch的平均速度`

⚪ fp32

| hardware | device | mobilenet_v3_small | mobilenet_v3_large | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | squeezenet1_1 | efficientnet_v2_s | vgg16_bn | resnet18 | resnet50 | densenet121 | resnext50_32x4d | convnext_tiny | vit_b_16 | vit_b_32 | swin_t | swin_v2_t |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| RTX3060 (GB@170W)     | cuda | 171.17 | 142.09 | 151.69 | 148.98 | 398.96 | 46.49 | 159.43 | 359.06 | 142.38 | 57.74 | 146.99 | 188.16 | 131.56 | 206.08 | 88.34 | 63.46 |
| RTX3060 (CF@100W LHR) | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40HX                  | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P104-100 (PCIe x1)    | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R780M                 | rocm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| U5 235H               | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 需绑核排除小小核
| i5-12400F             | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| i7-11700K             | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E5-2683v4             | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R9 8945HS             | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R5 1600X              | cpu  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

⚪ fp16

| hardware | device | mobilenet_v3_small | mobilenet_v3_large | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | squeezenet1_1 | efficientnet_v2_s | vgg16_bn | resnet18 | resnet50 | densenet121 | resnext50_32x4d | convnext_tiny | vit_b_16 | vit_b_32 | swin_t | swin_v2_t |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| RTX3060 (GB@170W)     | cuda | 159.93 | 127.09 | 127.28 | 128.64 | 393.79 | 46.02 | 298.07 | 314.79 | 127.78 | 51.99 | 128.27 | 189.04 | 213.36 | 208.72 | 91.29 | 65.40 |
| RTX3060 (CF@100W LHR) | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40HX                  | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P104-100 (PCIe x1)    | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R780M                 | rocm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

⚪ bf16

| hardware | device | mobilenet_v3_small | mobilenet_v3_large | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | squeezenet1_1 | efficientnet_v2_s | vgg16_bn | resnet18 | resnet50 | densenet121 | resnext50_32x4d | convnext_tiny | vit_b_16 | vit_b_32 | swin_t | swin_v2_t |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| RTX3060 (GB@170W)     | cuda | 148.39 | 120.39 | 123.07 | 121.50 | 382.50 | 44.13 | 290.97 | 314.87 | 125.02 | 50.14 | 125.52 | 186.93 | 211.48 | 198.56 | 88.49 | 63.76 |
| RTX3060 (CF@100W LHR) | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 40HX                  | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P104-100 (PCIe x1)    | cuda |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| R780M                 | rocm |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
