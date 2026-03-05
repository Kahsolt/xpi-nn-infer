#!/usr/bin/env python3

# 跑TTS-ASR循环

import warnings
warnings.filterwarnings(category=UserWarning, action='ignore')
warnings.filterwarnings(category=FutureWarning, action='ignore')

from time import time
from xpi_nn_infer.models.transformers import Whisper, KokoroTTS


tts = KokoroTTS()
asr = Whisper()

text = '这是一个中文测试'
for i in range(3):
  wav = tts.infer(text)
  text = asr.infer(wav, tts.sample_rate)
  print('text:', text)
print()

texts = [
  'Kokoro 是一系列体积虽小但功能强大的 TTS 模型。',
  '该模型是经过短期训练的结果，从专业数据集中添加了100名中文使用者。',
  '中文数据由专业数据集公司「龙猫数据」免费且无偿地提供给我们。感谢你们让这个模型成为可能。',
  '另外，一些众包合成英语数据也进入了训练组合：',
  '1小时的 Maple，美国女性。',
  '1小时的 Sol，另一位美国女性。',
  '和1小时的 Vale，一位年长的英国女性。',
  '由于该模型删除了许多声音，因此它并不是对其前身的严格升级，但它提前发布以收集有关新声音和标记化的反馈。',
  '除了中文数据集和3小时的英语之外，其余数据都留在本次训练中。',
  '目标是推动模型系列的发展，并最终恢复一些被遗留的声音。',
  '美国版权局目前的指导表明，合成数据通常不符合版权保护的资格。',
  '由于这些合成数据是众包的，因此模型训练师不受任何服务条款的约束。',
  '该 Apache 许可模式也符合 OpenAI 所宣称的广泛传播 AI 优势的使命。',
  '如果您愿意帮助进一步完成这一使命，请考虑为此贡献许可的音频数据。',
]

for text in texts:
  ts_start = time()
  wav = tts.infer(text)
  trans = asr.infer(wav, tts.sample_rate)
  ts_end = time()

  print('trans:', trans, f'({ts_end - ts_start:.3f}s)')
