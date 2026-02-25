#!/usr/bin/env python3

import xpi_nn_infer as XNI
import xpi_nn_infer.models.providers as P

XNI.get_default_io_backends()
{
  'image': '',
  'video': '',
  'audio': '',
  'camera': '',
  'micphone': '',
}
XNI.set_default_io_backend('liborosa')
XNI.set_default_io_backend('liborosa', type='audio')


def run():
  # prepare data
  dataset = get_dataset()
  print(dataset.path)
  print(dataset.type)
  print(dataset.n_samples)

  # init model & ckpt
  # 1. from known providers (auto download or local)
  model = P.torchvision.get_model('resnet18')
  model = P.torchvision.get_model('path/to/resnet18.pth')
  print(model.backend)
  print(model.path)
  # 2. from local
  model = P.get_model('resnet18')
  print(model.backend)
  print(model.path)

  # run infer
  model.infer(dataset, batch_size)
  # use callback_fn to process the model ouputs
  model.infer(dataset, batch_size, callback_fn)

  # convert to other backends
  ox_model = model.to_onnx()
  ox_model = OnnxModel.from_torch(model)
  ox_model.save(fp)

  # run infer again
  ox_model.infer(dataset, batch_size)
  ox_model.infer(dataset, batch_size, callback_fn)

  # compare...


if __name__ == '__main__':
  pass
