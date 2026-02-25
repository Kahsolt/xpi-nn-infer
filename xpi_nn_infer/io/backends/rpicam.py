#!/usr/bin/env python3

from picamera2 import Picamera2

# Picamera2 doc: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
# Camera hardware: https://www.raspberrypi.com/documentation/accessories/camera.html#hardware-specification
# Camera software: https://www.raspberrypi.com/documentation/computers/camera_software.html


def get_available_modes():
  '''
  [设备型号]
    兼容 Camera Module v1 (5-megapixel) sensor OV5647
  [通用参数]
    bit_depth: 10
    unpacked: SGBRG10
    format: SGBRG10_CSI2P
  [模式支持]
    | mode | size | fps target/real | crop_limits |
    | :-: | :-: | :-: | :-: |
    | 0 |  (640,  480) | 58.92 / 63.4 |  (16,   0, 2560, 1920) |
    | 1 | (1296,  972) | 46.34 / 46.8 |   (0,   0, 2592, 1944) |
    | 2 | (1920, 1080) | 32.81 / 33.0 | (348, 434, 1928, 1080) |
    | 3 | (2592, 1944) | 15.63 / 15.7 |   (0,   0, 2592, 1944) |
  [图像格式]
    - XBGR8888 - every pixel is packed into 32-bits, with a dummy 255 value at the end, so a pixel would look like [R, G, B, 255] when captured in Python.
    - XRGB8888 - as above, with a pixel looking like [B, G, R, 255].
    - RGB888 - 24 bits per pixel, ordered [B, G, R].
    - BGR888 - as above, but ordered [R, G, B].
    - YUV420 - YUV images with a plane of Y values followed by a quarter plane of U values and then a quarter plane of V values.
  '''

  cam = Picamera2()
  modes = cam.sensor_modes
  cam.close()
  for i, mode in enumerate(modes):
    print(f'[mode-{i}]')
    print(mode)
  return modes


def get_camera(mode:int=0, fps:int=30, buffers:int=3, format:str='RGB888') -> Picamera2:
  cam = Picamera2()
  mode = cam.sensor_modes[mode]
  config = cam.create_still_configuration(
    main={
      'size': mode['size'],
      'format': format,
    },
    sensor={
      'output_size': mode['size'],
      'bit_depth': mode['bit_depth'],
    },
    buffer_count=buffers,
  )
  cam.configure(config)
  cam.set_controls({'FrameRate': fps})
  return cam
