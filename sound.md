## 메타의 Next Level 음성 to 음성 번역 시스템 발표
음성에서 바로 음성으로 번역하는 번역 시스템의 한단계 진보한 기술인 Seamless 시리즈를 메타에서 발표하였습니다.
구글 Gemini 의 발표에 뭍혀 회자가 잘 안되고 있지만, 대단한 발전을 이룬 기술인데요.
사용자의 음성 스타일로, 감정 표현의 늬앙스까지 반영해서 음성으로 바로 번역 해 주는 SeamlessExpressive, 그리고  2초 이내의 준 실시간으로 음성 번역을 해 주는 SeamlessStreaming 을 발표했습니다.
첨부 영상은 영어로 속삭이듯이 말하는 것을 스페인어 음성 출력으로 변환한 예 입니다.
번역은 SeamlessM4T를 기반으로 하여, 입력언어는 약 100개, 출력 언어는 36개를 지원합니다. 세계적인 대세 언어답게 한국어도 지원합니다. 🤗 (한국어가 대세라니 한글 텍스트 입출력도 안되던 시대가 얼마전인 것 같은데 격세지감이네요. 😭)
메타 공식 사이트 👉 https://ai.meta.com/research/seamless-communication/
전세계를 연결하는 메타답게 오래전부터 번역 기술에 많은 노력을 기울이고 있는데요. 근데 왜 아직도 페이스북이나 인스타, 스레드에는 제대로 지원을 안해주고 있는지.. 😅 빨리 메타의 플랫폼에 적용되어 언어의 장벽 없이 전 세계 사람들과 쉽게 교류 할 날이 왔으면 좋겠습니다.

## 텍스트(Text-to-Audio)를 기준으로 일반 오디오를 합성해서 만들어주는 "AudioGen: Textually Guided Audio Generation"도 등장했네요. 
"남자가 키보드로 타이핑하면서 말하고 있다" 이렇게만 입력하면 이런 소리가 만들어진다는게 놀랍네요. 
paper: https://felixkreuk.github.io/text2audio_arxiv.../paper.pdf
sample: https://felixkreuk.github.io/text2audio_arxiv_samples/

## 화자 분리를 하는 관련 연구
Speaker Recognition: https://paperswithcode.com/task/speaker-recognition
Speaker Verification: https://paperswithcode.com/task/speaker-verification, https://paperswithcode.com/.../text-independent-speaker..., https://paperswithcode.com/.../text-dependent-speaker...
Speaker Identification: https://paperswithcode.com/task/speaker-identification
Speaker Separation: https://paperswithcode.com/task/speaker-separation,
Speaker Profiling: https://paperswithcode.com/task/speaker-profiling

## 음성인식 colab 실습 코드 공유
colab link: https://bit.ly/3qYVQeC

## SMART-Single_Emotional_TTS: https://github.com/SMART-TTS/SMART-Single_Emotional_TTS
Variable-length style embedding을 추출하여 반영하는 Unsupervised Style TTS 모델
SMART-Multi-Speaker-Style-TTS: https://github.com/SMART-TTS/SMART-Multi-Speaker-Style-TTS
VITS 모델 기반의 Multi-Speaker Style TTS 모델
SMART-G2P: https://github.com/SMART-TTS/SMART-G2P
영어, 한문 등을 포함한 한국어 문장을 위한 발음열 변환 모델
SMART-Vocoder: https://github.com/SMART-TTS/SMART-Vocoder
Variational inference 학습 기법을 이용한 다화자 보코더
SMART-Long_Sentence_TTS: https://github.com/SMART-TTS/SMART-Long_Sentence_TTS
Curriculum learning을 활용한 document-level 한국어 음성합성 모델
SMART-NAR_Fast_TTS: https://github.com/SMART-TTS/SMART-NAR_Fast_TTS
FastSpeech 모델을 기반으로 alignment를 external duration label 없이 학습하는 한국어 음성합성 모델
