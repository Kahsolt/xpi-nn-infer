#!/usr/bin/env python3

import os

import torch
from numpy import ndarray

from xpi_nn_infer.utils import exit_missing_packages

device = 'cuda' if torch.cuda.is_available() else 'cpu'

HF_ENDPOINT = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = HF_ENDPOINT

# Mainly focus on tiny/small models (1~4G RAM runnable)
HF_MODELS = {
  'llm': [
    'Qwen/Qwen3.5-0.8B',                            # 1.75 GB
    'Qwen/Qwen3-0.6B',                              # 1.5 GB; has GGUF versions
    'Qwen/Qwen2.5-0.5B-Instruct',                   # 988 MB; has GGUF versions
    'Qwen/Qwen2-0.5B-Instruct',                     # 988 MB; has GGUF versions
    'Qwen/Qwen2-0.5B-Instruct-AWQ',                 # 731 MB
    'Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4',           # 732 MB

    'OpenGVLab/InternVL3_5-1B',                     # 2.12 GB
    'OpenGVLab/InternVL3_5-1B-HF',                  # 2.12 GB
    'OpenGVLab/InternVL3_5-1B-Flash',               # 2.3 GB

    'internlm/internlm2_5-1_8b',                    # 3.78 GB

    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',           # 2.2 GB

    'HuggingFaceTB/SmolLM2-360M-Instruct',          # 724 MB
    'HuggingFaceTB/SmolLM2-360M',                   # 724 MB
    'HuggingFaceTB/SmolLM2-135M-Instruct',          # 269 MB
    'HuggingFaceTB/SmolLM2-135M',                   # 269 MB
    'HuggingFaceTB/SmolLM-360M-Instruct',           # 1.45 GB
    'HuggingFaceTB/SmolLM-360M',                    # 1.45 GB
    'HuggingFaceTB/SmolLM-135M-Instruct',           # 538 MB
    'HuggingFaceTB/SmolLM-135M',                    # 538 MB

    'tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct',   # 216 MB
    'tiiuae/Falcon-H1-Tiny-90M-Instruct',                 # 182 MB; has GGUF versions
    'tiiuae/Falcon-H1-Tiny-Coder-90M',                    # 182 MB
  ],
  'vllm': [
    'deepseek-ai/Janus-1.3B',                       # 4.18 GB
    'deepseek-ai/Janus-Pro-1B',                     # 4.18 GB
    'deepseek-ai/JanusFlow-1.3B',                   # 4.09 GB
    'deepseek-ai/deepseek-vl-1.3b-chat',            # 3.95 GB

    'Qwen/Qwen3-VL-2B-Thinking',                    # 4.26 GB
    'Qwen/Qwen3-VL-2B-Instruct',                    # 4.26 GB
    'Qwen/Qwen3-VL-Embedding-2B',                   # 4.26 GB

    'HuggingFaceTB/SmolVLM-Instruct-DPO',           # 4.49 GB
    'HuggingFaceTB/SmolVLM-Instruct',               # 4.49 GB
    'HuggingFaceTB/SmolVLM-Synthetic',              # 4.49 GB
    'HuggingFaceTB/SmolVLM-Base',                   # 4.49 GB
  ],
  'asr': [
    'Qwen/Qwen3-ASR-1.7B',                          # 4.69 GB
    'Qwen/Qwen3-ASR-0.6B',                          # 1.88 GB
    'Qwen/Qwen3-ForcedAligner-0.6B',                # 1.84 GB

    'FireRedTeam/FireRedASR2-LLM',                  # 3.63 GB
    'FireRedTeam/FireRedASR2-AED',                  # 4.73 GB
    'FireRedTeam/FireRedVAD',                       # 2.37 MB
    'FireRedTeam/FireRedLID',                       # 5.55 GB
    'FireRedTeam/FireRedPunc',                      # 407 MB

    'zai-org/GLM-ASR-Nano-2512',                    # 4.52 GB

    'openai/whisper-large-v3-turbo',                # 1.62 GB
    'openai/whisper-medium',                        # 3.06 GB
    'openai/whisper-small',                         # 967 MB
    'openai/whisper-base',                          # 290 MB
    'openai/whisper-tiny',                          # 151 MB

    'FunAudioLLM/Fun-ASR-Nano-2512',                # 1.97 GB
    'FunAudioLLM/Fun-ASR-MLT-Nano-2512',            # 1.97 GB
  ],
  'tts': [
    'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',         # 3.83 GB
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',         # 3.83 GB
    'Qwen/Qwen3-TTS-12Hz-1.7B-Base',                # 3.83 GB
    'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',         # 1.81 GB
    'Qwen/Qwen3-TTS-12Hz-0.6B-Base',                # 1.83 GB

    'FireRedTeam/FireRedTTS',                       # 2.12 GB

    'zai-org/GLM-TTS',                              # 7+ GB (!)

    'fishaudio/s1-mini',                            # 1.74 GB
    'fishaudio/fish-speech-1.5',                    # 1.28 GB
    'fishaudio/fish-speech-1.4',                    # 989 MB
    'fishaudio/fish-speech-1.2-sft',                # 981 MB
    'fishaudio/fish-speech-1.2',                    # 981 MB

    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',         # 3.3+ GB
    'FunAudioLLM/CosyVoice2-0.5B',                  # 2.5+ GB
    'FunAudioLLM/CosyVoice-300M-Instruct',          # 1.6+ GB
    'FunAudioLLM/CosyVoice-300M-SFT',               # 1.6+ GB
    'FunAudioLLM/CosyVoice-300M',                   # 1.6+ GB

    'hexgrad/Kokoro-82M-v1.1-zh',                   # 327 MB
    'hexgrad/Kokoro-82M',                           # 327 MB

    'Aratako/Irodori-TTS-500M',                     # 1.98 GB
    'Aratako/MioTTS-2.6B',                          # 4.98 GB
    'Aratako/MioTTS-1.7B',                          # 3.49 GB
    'Aratako/MioTTS-1.2B',                          # 2.39 GB
    'Aratako/MioTTS-0.6B',                          # 1.22 GB
    'Aratako/MioTTS-0.4B',                          # 733 MB
    'Aratako/MioTTS-0.1B',                          # 229 MB
  ],
  'ocr': [
    'FireRedTeam/FireRed-OCR',                      # 4.26 GB

    'zai-org/GLM-OCR',                              # 2.65 GB
  ],
}


class Qwen:

  def __init__(self, model_path:str='Qwen/Qwen3.5-0.8B', max_new_tokens:int=128):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    self.model_path = model_path
    self.max_new_tokens = max_new_tokens
    self.processor = AutoProcessor.from_pretrained(model_path)
    self.model = AutoModelForImageTextToText.from_pretrained(model_path)

  def infer(self, prompt:str) -> str:
    messages = [
      {
        'role': 'user',
        'content': [
          #{'type': 'image', 'url': 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG'},
          {'type': 'text', 'text': prompt}
        ]
      },
    ]
    inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(self.model.device)
    outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
    text = self.processor.decode(outputs[0][inputs['input_ids'].shape[-1]:])
    return text


class Falcon_H1_Tiny:

  '''
  https://huggingface.co/collections/tiiuae/falcon-h1-tiny
  https://huggingface.co/spaces/tiiuae/tiny-h1-blogpost
  '''

  def __init__(self, model_path:str='tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct', max_new_tokens:int=128):
    from transformers import AutoProcessor, AutoModelForCausalLM

    self.model_path = model_path
    self.max_new_tokens = max_new_tokens
    self.processor = AutoProcessor.from_pretrained(model_path)
    self.model = AutoModelForCausalLM.from_pretrained(model_path)

  def infer(self, prompt:str) -> str:
    messages = [
      {
        'role': 'user',
        'content': [
          {'type': 'text', 'text': prompt}
        ]
      },
    ]
    inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(self.model.device)
    outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
    text = self.processor.decode(outputs[0][inputs['input_ids'].shape[-1]:])
    return text


class QwenASR:

  def __init__(self, model_path:str='Qwen/Qwen3-ASR-0.6B', language:str='Chinese'):
    try:
      from qwen_asr import Qwen3ASRModel
    except ImportError:
      exit_missing_packages(['qwen-asr'])

    self.model_path = model_path
    self.language = language
    self.model = Qwen3ASRModel.from_pretrained('Qwen/Qwen3-ASR-1.7B', device_map=device,
      max_inference_batch_size=1,  # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
      max_new_tokens=256,          # Maximum number of tokens to generate. Set a larger value for long audio input.
    )

  def infer(self, wav:str, language:str=None) -> str:
    result = self.model.transcribe(
        audio='https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav',
        language=self.language or language,
    )[0]
    return result.text


class QwenTTS:

  def __init__(self, model_path:str='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'):
    try:
      from qwen_tts import Qwen3TTSModel
    except ImportError:
      exit_missing_packages(['qwen-tts'])

    self.model_path = model_path
    self.model = Qwen3TTSModel.from_pretrained(model_path, device_map=device).eval()
    self.sample_rate = self.model.generate_custom_voice(text='测试模型输出的采样率', language='Chinese')[1]

  @torch.inference_mode()
  def infer(self, text:str, language:str='Chinese', speaker:str='Vivian', prompt:str='用平静的语气说') -> str:
    wavs, sr = self.model.generate_custom_voice(text=text, language=language, speaker=speaker, instruct=prompt)
    assert sr == self.sample_rate
    return wavs[0]


class SmolLM:

  def __init__(self, model_path:str='HuggingFaceTB/SmolLM2-135M-Instruct'):
    self.model_path = model_path


class Whisper:

  def __init__(self, model_path:str='openai/whisper-tiny', prompt:str='输出简体中文'):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    self.model_path = model_path
    self.processor = WhisperProcessor.from_pretrained(model_path)
    self.model = WhisperForConditionalGeneration.from_pretrained(model_path).eval()
    self.model.config.forced_decoder_ids = None
    # https://github.com/huggingface/transformers/issues/23845
    self.prompt_ids = self.processor.get_prompt_ids(prompt, return_tensors='pt')
    self.sample_rate = 16000

  @torch.inference_mode()
  def infer(self, wav:ndarray, sr:int=16000) -> str:
    if sr != self.sample_rate:
      from scipy import signal
      wav = signal.resample(wav, round(len(wav) / sr * self.sample_rate))
    input_features = self.processor(wav, sampling_rate=self.sample_rate, return_tensors='pt').input_features
    predicted_ids = self.model.generate(input_features, prompt_ids=self.prompt_ids, language='chinese')[0]
    transcription = self.processor.decode(predicted_ids, skip_special_tokens=False)
    return transcription


class KokoroTTS:

  def __init__(self, model_path:str='hexgrad/Kokoro-82M-v1.1-zh', voice:str='zf_001'):
    try:
      from kokoro import KModel, KPipeline
    except ImportError:
      exit_missing_packages(['kokoro>=0.8.2', '"misaki[zh]>=0.8.2"'])

    self.model_path = model_path
    self.voice = voice  # https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/tree/main/voices
    self.sample_rate = 24000
    self.model = KModel(repo_id=model_path).to(device).eval()
    self.pipeline = KPipeline(lang_code='z', repo_id=model_path, model=self.model)

  # HACK: Mitigate rushing caused by lack of training data beyond ~100 tokens
  # Simple piecewise linear fn that decreases speed as len_ps increases
  def speed_callable(self, len_ps:int):
    speed = 0.8
    if len_ps <= 83:
      speed = 1
    elif len_ps < 183:
      speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

  @torch.inference_mode()
  def infer(self, text:str, voice:str=None) -> ndarray:
    generator = self.pipeline(text, voice=voice or self.voice, speed=self.speed_callable)
    result = next(generator)
    return result.audio


class CosyVoice:

  '''https://github.com/FunAudioLLM/CosyVoice'''

  def __init__(self, model_path:str='FunAudioLLM/Fun-CosyVoice3-0.5B-2512'):
    try:
      from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError:
      exit_missing_packages(["cosyvoice"])

    self.model_path = model_path
    self.model = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    self.sample_rate = self.model.sample_rate

  @torch.inference_mode()
  def infer(self, text:str, prompt:str=None) -> ndarray:
    if prompt:
      prompt = f'You are a helpful assistant.{prompt}<|endofprompt|>'
      result = self.model.inference_instruct2(text, prompt, './asset/zero_shot_prompt.wav', stream=False)
    else:
      prompt = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
      result = self.model.inference_zero_shot(text, prompt, './asset/zero_shot_prompt.wav', stream=False)
    return result['tts_speech']


class MioTTS:

  '''Support only English & Japanese'''

  def __init__(self, model_path:str='Aratako/MioTTS-0.1B'):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    self.model_path = model_path
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.model = AutoModelForCausalLM.from_pretrained(model_path).eval()

  @torch.inference_mode()
  def infer(self, text:str) -> ndarray:
    input_ids = self.tokenizer(text, return_tensors='pt')['input_ids']
    result = self.model.generate(input_ids)[0]
    # TODO: ??? 这个模型咋用啊，输出是一堆id啊，这不是TTS模型吗？
    return result
