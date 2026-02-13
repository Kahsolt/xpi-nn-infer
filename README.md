# xpi-nn-infer

    A lightweight toolbox for NN model convert and infer on ARM SoC devices like RaspberryPi, OrangePi, LubanCat, etc.

----

### Quickstart

```
pip install xpi-nn-infer
python -m xpi_nn_infer.tools.install_default_backends
```

### Backends

ℹ xpi-nn-infer is a bare framework, you need configure backends to make it work.

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
| torch    |  |  |
| tflite   |  |  |
| paddle   |  |  |
| onnx     |  |  |
| openvino |  |  |
| ncnn     |  |  |

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
