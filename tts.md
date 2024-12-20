# awesome-tts-samples

List of TTS papers with audio samples provided by the authors. 
The last rows of each paper show the spectrogram inversion (vocoder) being used.

## 2021 
- [wavegad2](https://github.com/mindslab-ai/wavegrad2)

## 2020
- [Glow-TTS](https://arxiv.org/abs/2005.11129) - Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search
  - https://jaywalnut310.github.io/glow-tts-demo
  - WaveGlow
- [Flowtron](https://arxiv.org/abs/2005.05957) - Flowtron: an Autoregressive Flow-based Generative Network for Text-to-Speech Synthesis
  - https://nv-adlr.github.io/Flowtron
  - WaveGlow

## 2019
- [Tacotron2+DCA](https://arxiv.org/abs/1910.10288) - Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis
  - https://google.github.io/tacotron/publications/location_relative_attention
  - WaveRNN
- [GAN-TTS](https://openreview.net/forum?id=r1gfQgSFDr) - High Fidelity Speech Synthesis with Adversarial Networks
  - https://storage.googleapis.com/deepmind-media/research/abstract.wav
  - End-to-end model (Built on top of 200Hz linguistic & log pitch features)
- [Multi-lingual Tacotron2](https://arxiv.org/abs/1907.04448) - Learning to Speak Fluently in a Foreign Language: Multilingual Speech Synthesis and Cross-Language Voice Cloning
  - https://google.github.io/tacotron/publications/multilingual
  - WaveRNN
- [MelNet](https://arxiv.org/abs/1906.01083) - MelNet: A Generative Model for Audio in the Frequency Domain
  - https://audio-samples.github.io
  - https://sjvasquez.github.io/blog/melnet
  - [Gradient-based spectrogram inversion](https://gist.github.com/carlthome/a4a8bf0f587da738c459d0d5a55695cd)
- [FastSpeech](https://arxiv.org/abs/1905.09263) - FastSpeech: Fast, Robust and Controllable Text to Speech
  - https://speechresearch.github.io/fastspeech
  - WaveGlow
- [ParaNet](https://arxiv.org/abs/1905.08459) - Parallel Neural Text-to-Speech
  - https://parallel-neural-tts-demo.github.io
  - WaveVAE, ClariNet, WaveNet

## 2018
- [Transformer-TTS](https://arxiv.org/abs/1809.08895) - Neural Speech Synthesis with Transformer Network
  - https://neuraltts.github.io/transformertts
  - WaveNet
- [Multi-speaker Tacotron2](https://arxiv.org/abs/1806.04558) - Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis
  - https://google.github.io/tacotron/publications/speaker_adaptation
  - WaveNet
- [Tacotron2+GST](https://arxiv.org/abs/1803.09017) - Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis
  - https://google.github.io/tacotron/publications/global_style_tokens
  - Griffin-Lim

## 2017
- [Tacotron2](https://arxiv.org/abs/1712.05884) - Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions
  - https://google.github.io/tacotron/publications/tacotron2
  - WaveNet
- [Tacotron](https://arxiv.org/abs/1703.10135) - Tacotron: Towards End-to-End Speech Synthesis
  - https://google.github.io/tacotron/publications/tacotron
  - Griffin-Lim

주제 : Tocotron + Wavenet을 이용한 한국어 TTS 구현
Tocotron + Wavenet Vocoder + Korean TTS
https://github.com/hccho2/Tacotron-Wavenet-Vocoder

Tacotron에 대해 쉽게 이해할 수 있도록 DEVIEW 2017에서 발표한 영상
https://tv.naver.com/v/2292650

wavenet 및 기타 합성 음성에 대한 설명
https://cloud.google.com/text-to-speech/docs/wavenet

google cloud text-to-speech
https://cloud.google.com/text-to-speech/

tensorflow Simple Audio Recognition
https://www.tensorflow.org/tutorials/sequences/audio_recognition

2) 가용데이터 유무, 데이터 사이즈 (kaggle 및 
Korean Single Speaker Speech Dataset
https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset
Audio File Type: wav
Total Running Time: 12+ hours
Sample Rate: 44,100 KHZ
Number of Audio Files: 12,853
오디오의 침묵구간을 기준으로 분리되어 있고, 내용에 대한 라벨링이 되어 있어 좋을것 같습니다.

Custom Datasets의 제작은 https://github.com/carpedm20/multi-speaker-tacotron-tensorflow 에서 자세히 설명하고 있으나, 
Google Speech Recognition API을 이용하여 audio->text로 변환하는 과정에서 정확하지 않은 데이터가 발생하여 하나씩 들어보고
수정해야 할 것 같습니다.

3) 해당데이터를 이용한 최신 성능 (SOTA) 기재 
google cloud text-to-speech
https://cloud.google.com/text-to-speech/
다양한 언어의 텍스트를 다양한 음성으로 Wavenet 기반기술을 사용하여 인간과 유사한 목소리로 합성
끊어 읽을 지점, 숫자, 날짜, 시간 형식, 기타 발음 지침을 추가할 수 있는 SSML 태그로 음성을 맞춤설정가능

  - 본인이 해당 데이터로 해보려는 세부 task를 명확하게 명시 필요
Tocotron + Wavenet을 이용한 한국어 TTS 구현
1) Multi-Speaker Tacotron의 한국어 구현
2) Wavenet vocoder (r9y9)

  - Reference 논문/웹사이트
Tacotron
https://arxiv.org/abs/1703.10135
wavenet 논문
https://arxiv.org/abs/1609.03499
딥러닝 (Tacotron, wavenet)에 대한 이해
https://github.com/hccho2/hccho2.github.io/blob/master/DeepLearning.pdf
딥러닝을 이용한 음성합성 관련 자료 모음
https://github.com/lifefeel/SpeechSynthesis
딥 러닝을 이용한 자연어 처리 입문
https://wikidocs.net/book/2155

4) 데이터를 활용한 오픈소스 링크 기재 
Tocotron + Wavenet Vocoder + Korean TTS
https://github.com/hccho2/Tacotron-Wavenet-Vocoder

wavenet의 구현
A TensorFlow implementation of DeepMind's WaveNet paper
https://github.com/ibab/tensorflow-wavenet

Tocotron의 한국어 구현
https://github.com/carpedm20/multi-speaker-tacotron-tensorflow

## 인간이 말하는 음성 합성으로 텍스트를 읽는 TTS 모델 「StyleTTS 2」 콜롬비아대의 연구자들 개발
StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models (2311, Columbia University)
논문 https://arxiv.org/abs/2306.07691v2 (2311)
신형의 Text-to-Speech(TTS) 모델 「StyleTTS 2」는, 선행하는 「StyleTTS」를 기본으로 하고 있어, 보다 인간다운 음성 합성을 목표로 하고 있습니다.
StyleTTS 2는 음성 스타일을 선택하는 확산 모델과 대규모 음성 언어 모델(Large Speech Language Model, SLM)을 결합한 적대적인 교육을 채택합니다. 이렇게 하면 참조 음성이 없어도 텍스트에 적합한 스타일의 음성을 자동으로 생성할 수 있어 다양한 음성 유형에 대응합니다.
또한 Wav2Vec 2.0, HuBERT, WavLM과 같은 대규모 사전 훈련된 SLM이 사용되며, 이러한 조합은 합성된 음성의 자연성을 향상시키고 보다 인간적인 느낌을 제공합니다.
StyleTTS 2에 의해 생성된 음성의 평가 결과는 다음과 같다. LJSpeech 데이터 세트는 네이티브 영어 화자의 평가로 인간의 녹음을 능가하는 점수를 받았습니다. 또한 VCTK 데이터 세트는 자연스러움과 참조 화자 간의 유사성 모두에서 인간 수준의 성능을 보여주었습니다.
게다가 최첨단 기술인 'NaturalSpeech'에 비해 더 높은 점수를 얻었습니다. LibriTTS 데이터세트에서의 교육은 이전 공개 모델을 넘어서는 자연스러움을 보여주었으며, 데이터 양이 적은 업적을 달성했습니다.

