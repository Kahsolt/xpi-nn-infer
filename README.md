# xpi-nn-infer

    A lightweight toolbox for NN model convert and infer on ARM SoC devices like RaspberryPi, OrangePi, LubanCat, etc.

----

### Installation

You can either install from PyPI:

```shell
# create venv (optional but recommended!!)
conda create -n xpi python==3.13    # py3.10 or later
conda activate xpi
# install framework
pip install xpi-nn-infer
# install default backends
python -m xpi_nn_infer.tools.install_default_backends -y
```

or install locally:

```shell
# create venv (optional but recommended!!)
conda create -n xpi python==3.13    # py3.10 or later
conda activate xpi
# clone this repo
git clone https://gitee.com/kahsolt/xpi-nn-infer.git
cd xpi-nn-infer
# install locally
pip install -e .
# install default backends
python -m xpi_nn_infer.tools.install_default_backends -y
```

### Usage

⚪ Use via command line

```shell
# benchmark with random inputs
python -m xpi_nn_infer.tools.benchmark_torchvision -K torch
python -m xpi_nn_infer.tools.benchmark_torchvision -K onnx     -M mobilenet_v3_small
python -m xpi_nn_infer.tools.benchmark_torchvision -K openvino -M mobilenet_v3_large              # cls
python -m xpi_nn_infer.tools.benchmark_torchvision -K openvino -M lraspp_mobilenet_v3_large       # seg
python -m xpi_nn_infer.tools.benchmark_torchvision -K openvino -M ssdlite320_mobilenet_v3_large   # det (but not supported yet :(

# infer from random inputs, image file or folder
python -m xpi_nn_infer.tools.infer_torchvision -K openvino -M resnet18 -I random
python -m xpi_nn_infer.tools.infer_torchvision -K openvino -M resnet18 -I path/to/your/image.jpg
python -m xpi_nn_infer.tools.infer_torchvision -K openvino -M resnet18 -I path/to/your/image_folder
```

⚪ Use via API

ℹ TODO! TODO!! TODO!!!

### Configurations

ℹ xpi-nn-infer is a bare framework, you need configure backends to make it work.

#### Envvars

- MODEL_PATH: folder path for auto-downloaded or converted model checkpoints, defaults to `<site-packages>/xpi_nn_infer/MODEL_PATH` (PyPI install) or `<xpi-nn-infer>/MODEL_PATH` (local install)

#### Model providers

Thanks to all the open-source model providers 🎉

| name | supported | comment |
| :-: | :-: | :-: |
| torchvision  | √ |  |
| ultralytics  |   |  |
| paddleocr    |   |  |
| transformers |   |  |
| diffusers    |   |  |
| modelscope   |   |  |

#### NN backends

| name | supported | comment |
| :-: | :-: | :-: |
| torch      | √ |  |
| tensorflow |   |  |
| tflite     |   |  |
| paddle     |   |  |
| paddlelite |   |  |
| onnx       | √ |  |
| openvino   | √ |  |
| ncnn       |   |  |
| mnn        |   |  |
| mace       |   |  |

#### IO backends (⚠ Work In Progress!!)

| name | supported | type | comment |
| :-: | :-: | :-: |
| pillow      | √ | img     |   |
| skimage     |   | img     |   |
| imageio     |   | img     |   |
| torchvision |   | img     |   |
| cv2         |   | img/cam |   |
| rpicam      |   | cam     |   |
| ffmpy       |   | vid     |   |
| moviepy     |   | vid     |   |
| wave        |   | aud     |   |
| soundfile   |   | aud     |   |
| sounddevice |   | mic     |   |
| pyaudio     |   | aud/mic |   |
| pydub       |   | aud     |   |
| librosa     |   | aud     |   |
| scipy       |   | aud     |   |

### Supported Devices

- BCM2837: RaspberryPi 3B
- H618: OrangePi Zero 3
- RK3576: LubanCat3
- RK3399: FMX1 Pro, MRK3399
