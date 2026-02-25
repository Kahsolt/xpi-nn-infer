# xpi-nn-infer

    A lightweight toolbox for NN model convert and infer on ARM SoC devices like RaspberryPi, OrangePi, LubanCat, etc.

----

### Quickstart

```
pip install xpi-nn-infer
python -m xpi_nn_infer.tools.install_default_backends -y
python -m xpi_nn_infer.tools.infer_torchvision -K torch -M resnet50 -I random
python -m xpi_nn_infer.tools.benchmark_torchvision -K openvino -M resnet50
```

### Supported Devices

- BCM2837: RaspberryPi 3B
- H618: OrangePi Zero 3
- RK3576: LubanCat3
- RK3399: FMX1 Pro, MRK3399

### Configurations

ℹ xpi-nn-infer is a bare framework, you need configure backends to make it work.

#### Envvars

- MODEL_PATH: folder path for auto-downloaded or converted model checkpoints, defaults to `/xpi_nn_infer/MODEL_PATH`

#### Model providers

Thanks to all the open-source model providers 🎉

| name | supported | comment |
| :-: | :-: | :-: |
| torchvision  |  |  |
| ultralytics  |  |  |
| paddleocr    |  |  |
| transformers |  |  |
| diffusers    |  |  |
| modelscope   |  |  |

#### NN backends

| name | supported | comment |
| :-: | :-: | :-: |
| torch      |  |  |
| tensorflow |  |  |
| tflite     |  |  |
| paddle     |  |  |
| paddlelite |  |  |
| onnx       |  |  |
| openvino   |  |  |
| ncnn       |  |  |
| mnn        |  |  |
| mace       |  |  |

#### IO backends

| name | type | comment |
| :-: | :-: | :-: |
| pillow      | img     |  |
| skimage     | img     |  |
| imageio     | img     |  |
| torchvision | img     |  |
| cv2         | img/cam |  |
| rpicam      | cam     |  |
| ffmpy       | vid     |  |
| moviepy     | vid     |  |
| wave        | aud     |  |
| soundfile   | aud     |  |
| sounddevice | mic     |  |
| pyaudio     | aud/mic |  |
| pydub       | aud     |  |
| librosa     | aud     |  |
| scipy       | aud     |  |
