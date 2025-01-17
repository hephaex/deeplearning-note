## Hailuo Audio HD로 꿈의 목소리를 찾아보세요: 음성 합성의 새로운 시대!
MiniMax는 텍스트 음성 변환 기술의 차세대 혁신인 T2A-01-HD를 자랑스럽게 공개합니다. 이 모델은 탁월한 다용도성, 감성적 깊이, 다국어 진정성을 바탕으로 음성 합성의 가능성을 재정의합니다. 특별한 점은 다음과 같습니다.
[무한한 음성 사용자 지정] 
- 단 10초의 오디오로 모든 뉘앙스와 감정적 어조를 유지하면서 음성을 복제합니다.
- 언어, 성별, 악센트, 연령 및 스타일별로 분류된 300개 이상의 사전 구축된 음성 라이브러리에 액세스합니다.
- 고급 매개 변수 컨트롤로 피치, 속도 및 감정 톤을 사용자 정의하여 역동적인 결과를 얻을 수 있습니다.
- 실내 음향 및 전화 필터와 같은 전문 효과를 추가하여 스튜디오 수준의 결과물을 얻을 수 있습니다.
[정교한 감성 지능]
-음성의 미묘한 감정 뉘앙스를 포착하고 재현하는 업계 최초의 지능형 감성 시스템으로 음성에 생동감을 불어넣습니다.
-완벽한 음성 표현을 위해 자동 감정 감지 또는 수동 제어 중에서 선택할 수 있습니다.
[진정한 언어 전문성]
-17개 이상의 언어로 유창하게 말하며 지역적 진정성을 반영하는 자연스러운 억양으로 표현할 수 있습니다.
-지원 언어: 영어(미국, 영국, 호주, 인도)
 중국어(북경어 및 광동어)
 일본어, 한국어, 프랑스어, 독일어, 스페인어, 포르투갈어(브라질 포함), 이탈리아어, 아랍어, 러시아어, 터키어, 네덜란드어, 우크라이나어, 베트남어, 인도네시아어.
https://hailuo.ai/audio
## eSpeak NG - 100개 이상의 언어와 악센트를 지원하는 음성 합성기 오픈소스 (github.com/espeak-ng)
리눅스, 윈도우, 안드로이드 및 기타 OS
"Formant Synthesis(포먼트 합성)" 방식을 사용하는 eSpeak 엔진 기반
작은 크기로 많은 언어를 지원해서 윈도우 및 구글 번역 엔진 등에서 사용했었음(지금은 많이 자체엔진으로 대체됨)
음성은 선명하고 빠른 속도로 사용할 수 있지만 사람의 음성 녹음을 기반으로 하는 대형 합성기만큼 자연스럽거나 부드럽지는 않음
또한 Klatt 포먼트 합성을 지원하며, 백엔드 음성 합성기로 MBROLA를 사용가능
지원 형태
커맨드 라인 프로그램 : 리눅스 & 윈도우. 파일 및 Stdin 으로 받은 문자열 읽기
Shared 라이브러리(윈도우에서는 DLL)
윈도우용 SAPI5 버전. 스크린 리더 및 다른 프로그램에서 SAPI5 인터페이스로 이용 가능
Solaris, MacOS 를 포함한 다양한 플랫폼으로 이식
기능
특성을 변경할 수 있는 다양한 음성을 포함
음성 출력을 WAV 파일로 생성할 수 있음
SSML(음성 합성 마크업 언어)이 지원되며(완전하지 않음) HTML도 지원
컴팩트한 크기. 여러 언어를 포함한 프로그램과 데이터의 총 용량은 몇 MB 정도
MBROLA diphone Voices의 프런트엔드로 사용할 수 있음. eSpeak NG는 텍스트를 음높이와 길이 정보가 있는 음소로 변환
MBROLA는 음성 합성을 위한 음절 모음으로 구성된 오픈 소스 음성 엔진
음성은 상업적 목적이 아닌 경우 무료로 제공되지만 오픈 소스는 아님
텍스트를 음소 코드로 번역할 수 있으므로 다른 음성 합성 엔진의 프런트 엔드로 적용 가능
다른 언어에 대한 지원 추가 가능. 여러 언어가 다양한 단계로 포함되어 있음
C로 작성됨

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
