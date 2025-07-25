## MultiTalk: Audio-Driven Multi-Person Conversational Video Generation
MultiTalk은 오디오 기반 다중 인물 대화형 비디오 생성 기술. 이 기술은 다중 인물 대화, 노래, 상호작용 제어 그리고 애니메이션를 포함한 비디오 생성을 가능하게 함. 기존의 단일 인물이 아닌 다중 인물 대화형 비디오 생성을 가능하게 함. 오픈소스 공개(Apache-2.0 license)
MultiTalk, Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation (2505, Sun Yat-sen University, Meituan, HKUST)
project: https://meigen-ai.github.io/multi-talk/
paper: https://arxiv.org/abs/2505.22647
code: https://github.com/MeiGen-AI/MultiTalk
ComfyUI kijai wan base worflow : https://github.com/.../wanvideo_multitalk_test_02.json   
demo: https://huggingface.co/spaces/fffiloni/Meigen-MultiTalk


## 음악 생성 AI Suno, 
▲업그레이드된 곡 편집기(Song Editor) 
 - 곡 편집기는 사용자가 트랙의 각 파트를 직접 재배열하거나 재작성, 재구성할 수 있게 한다. 이전보다 정밀한 편곡과 편집을 가능하게 하는 기능.
   
▲스템 추출(Stem Extraction)
 - 스템(개별 악기) 추출 기능을 통해 노래를 보컬과 드럼, 베이스 등 12개의 개별 트랙으로 분리할 수 있으며, 각 파트를 미리 들어보고 개별적으로 다운로드 가능.

▲업로드 확대 
 - 최대 8분 길이의 전체 곡은 물론, 짧은 기타 리프나 사용자의 흥얼거림 등 샘플을 업로드. 이를 기반으로 새로운 창작을 시작할 수 있게 해줌.

▲크리에이티브 슬라이더(Creative Sliders) 
 - 결과물의 독창성, 구조적 정밀도, 참조 기반 정도를 세 가지 조절 항목으로 설정하는 기능. 예를 들어, 슬라이더를 조정해 사운드의 펑키함을 더하거나 줄이기 가능.

## Smoothie Qwen
최고의 성능을 기록한 Qwen 모델의 토큰 확률을 조정하는 후처리 도구
재학습 없이도 언어 출력의 편향을 줄여 모델의 성능을 높이는 경량 후처리 프로그램
과도한 중국어 발화 비율을 제어하여 뛰어난 성능을 자랑하는 Qwen 모델이 한국어를 더 잘 할 수 있도록 조정하는 프로그램
Qwen2.5 대응으로 개발을 시작하였으나 최근에 새롭게 나온 Qwen3도 동일하게 대응해 처리
미리 변환한 모델 링크
- Smoothie Qwen2.5 https://huggingface.co/.../smoothie-qwen25...
- Smoothie Qwen3 https://huggingface.co/.../smoothie-qwen3...
모든 라이센스는 Qwen의 라이센스를 그대로 따라 Apache 2.0으로 배포

- 공식 깃헙 https://github.com/dnotitia/smoothie-qwen
- 회사 공식 블로그 소개 https://medium.com/.../smoothie-qwen-smooth-out-your...
- Smoothie Qwen의 논문 : https://arxiv.org/abs/2507.05686


## Hunyuan3D-PolyGen   @TencentHunyuan
새롭게 업그레이드된 업계 최초의 아트급 3D 제너레이티브 모델인 Hunyuan3D-PolyGen을 소개합니다. 손쉬운 지능형 리토폴로지를 제공하여 전문 아트 파이프라인을 위한 AI 생성 모델을 제공

## Nanonets-OCR-s 
– 문서를 구조화된 마크다운으로 변환하는 OCR 모델
* 단순한 문자 인식 수준을 넘어 문서 전체를 Markdown 구조로 변환하는 고성능 이미지-to-Markdown OCR 모델
* 수학식은 LaTeX 형태로 변환하고, 이미지에는 자동 설명을 추가하며, 표는 HTML/Markdown 표로 출력해 LLM 활용에 최적화된 출력물을 생성
* 서명, 워터마크, 체크박스 등을 인식하여 <signature>, <watermark>, ☐/☑ 형태로 변환하는 등 문서 구성 요소별 처리 능력이 뛰어남
* Hugging Face의 Transformers 또는 vLLM 서버를 통해 손쉽게 활용 가능하며, docext 라이브러리를 통해 웹 앱 형태로도 사용 가능
* 다양한 문서 유형과 복잡한 레이아웃에 대해 정확도와 구조화 수준이 매우 높아, 계약서, 양식, 리포트 등에서 매우 유용함
* 

## ATI(Bytedance): Trajectory Instruction Video Generation ComyUI Test

중국 Bytedance에서 발표

이미지와 궤적 지시를 통하여 모션 영상 생성가능     
궤적을 이용함으로 텍스트로 지시하는 것보다는 더 정확하게 여러 모션 컨트롤을 만들어 내는게 가능

comfyui workflow : [https://github.com/.../examp.../wanvideo_ATI_testing_01.json](https://github.com/bytedance/ATI.git)

## LLM을 사용하여 위키백과 스타일의 아티클을 생성하는 STORM Research Assistant 를 만들었습니다.
STORM 리서치는 Stanford 에서 개발한 방법론으로, 처음부터 근거가 있고 체계적인 긴 형식의 기사를 작성하도록 설계되었습니다.

- 논문: Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models
- 논문 링크: https://arxiv.org/abs/2402.14207

이 프로젝트는 LangChain YouTube(https://youtu.be/1uUORSZwTz4?si=4RrM3UIuwwdWKFET) 에서도 영감을 얻었습니다. 작동 방식은 다음과 같습니다.
1. 주제를 정합니다(예: "암호학에서 양자 컴퓨팅의 미래").
2. 서로 다른 전문성을 가진 여러 AI 분석가를 생성합니다.
3. 각 분석가가 인터뷰 및 정보 검색을 수행합니다.
4. 마지막으로 모든 것을 잘 구조화된 문서로 결합합니다
5. 프로젝트: https://github.com/teddynote-lab/STORM-Research-Assistant


## 카카오, sLM ‘카나나 1.5’ 4종 오픈 소스 공개, "코딩·수학·도구사용 향상"
카카오는 23일 소형언어모델(SLM) '카나나(Kanana)' 1.5 버전 모델군을 오픈 소스로 공개. 
  - ▲ 카나나-1.5-8b-베이스(base) 
  - ▲ 카나나-1.5-8b-인스트럭트(instruct) 
  - ▲ 카나나-1.5-2.1b-베이스 
  - ▲ 카나나-1.5-2.1b-인스트럭트 4종.
  - 허깅페이스 공개.(아파치 2.0 라이선스)
- ▲카나나-1.5-8b-베이스(base)
- ▲카나나-1.5-8b-인스트럭트(instruct)
- ▲카나나-1.5-2.1b-베이스
- ▲카나나-1.5-2.1b-인스트럭트 등 4종. 
- 허깅페이스에서 다운받을 수 있으며, 아파치 2.0 라이선스를 적용해 수정과 상업적 활용이 가능.
- 지난해 10월 선보인 '카나나 나노'의 후속 버전. 가장 큰 변화는 기존 8000토큰에 불과했던 컨텍스트 창 크기를 12만8000토큰으로 확장.
- AI 에이전트 구현을 위해 코딩과 수학 문제 해결, 함수 호출 기능 강화에 중점을 뒀다고 소개. 벤치마크 결과, 이전 버전에 비해 코딩과 수학 문제 해결, 함수호출 능력에서 평균 1.5배의 성능 향상을 기록.
- 현재는 AI 에이전트를 목표로 '카나나2'를 개발 중. 이를 위해 ▲수학·코딩 관련 데이터를 수집하기 위한 파이프라인 구축 ▲컨텍스트 창 확장 ▲추론 특화 모델 및 하이브리드 모델 개발 등을 진행중.

## 구글, 코딩 에이전트 '줄스(Jules): '병렬 처리·비동기 방식'
구글이 I/O에서 가장 기대를 모았던 인공지능(AI) 코딩 에이전트를 공개. 

병렬 처리와 비동기 방식이 핵심. 

오픈AI도 지난 주말 '코덱스' 공개, 이것도 병렬 처리와 비동기 방식. 
- 구글은 20 열린 개발자 회의(I/O)를 통해  코딩 에이전트 '줄스(Jules)'를 테스트 버전으로 출시한다고 발표.
- 줄스는 지난해 12월 제미나이 2.0 출시 당시 처음 소개. 당시에는 AI 음성 비서 '프로젝트 아스트라'에 가렸으며, 소수 테스트에게만 공개된다고 밝혀 별로 주목받지 못했음.
- 이날부터는 베타 버전으로 출시, 대기자 명단 없이 바로 사용해 볼 수 있게 됨.
- 처음에는 무료로 사용량 제한이 있으며, 나중에 유료화될 예정.
- 줄스는 깃허브에 통합되며, '제미나이 2.5 프로'를 사용. 구글은 이번 행사에 앞서 코딩 성능이 대폭 향상된 I/O 버전 제미나이를 공개.
- 이 모델은 처음으로 앤트로픽 '클로드'의 코딩 벤치마크 성적을 넘어서 화제.
- 일부 사용자들은 이미 성능에 호평. 로렌스 버클리 국립연구소의 연구원들은 줄스와 관련 AI 도구를 사용하여 특정 분석 작업을 일주일에서 몇분으로 단축했다고 밝혀.
- 구글은 단순히 수정 사항만 제안하는 기존 코딩 어시스턴트와 달리, 줄스가 코드베이스를 분석하고 포괄적인 수정 계획을 수립하고 여러 파일에 대한 수정을 동시에 실행한다고 강조.
- "소프트웨어 개발에 대한 접근 방식을 근본적으로 바꾸는 것"
- 줄스도 병렬 처리와 비동기 방식이 핵심.
- 그러나 이는 지난주부터 오픈AI와 MS가 이미 다 공개한 사항. 여기에 구글은 줄스 외에도 코드 어시스트(Code Assist)와 AI 스튜디오(AI Studio), 파이어베이스(Firebase) 등의 코딩 도구를 이미 제공
- 구글은 또 하나의 에이전트인 '프로젝트 마리너'도 정식 출시. 이는 이날 새로 추가된 월 250달러짜리 '울트라' 요금제 사용자에게 제공.
- 마리너 역시 지난해 12월 처음 공개. 앤트로픽의 '컴퓨터 유즈'나 오픈AI의 '오퍼레이터'처럼 웹 브라우저를 인간 대신 조작, 온라인 쇼핑을 해주거나 비행기 티겟을 구매해 줄 수 있음.
- 마리너에도 병렬 처리와 비동기 방식을 적용. 즉, 새로운 마리너는 최대 10개의 작업을 동시에 진행할 수 있으며, 사용자는 마리너가 백그라운드에서 작업을 완료하는 동안 다른 프로젝트를 진행 가능.
- 오픈AI는 이번 행사의 하이라이트 중 하나인 에이전트 발표에 앞서 지난 주말 '코덱스'를 공개

## Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures
Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan Huang, Shangyan Zhou, Shirong Ma, Wenfeng Liang, Ying He, Yuqing Wang, Yuxuan Liu, Y.X. Wei

DeepSeek AI 2025

https://arxiv.org/html/2505.09343v1 

## 엔비디아, 오픈 소스 최고 성능 '코드 추론' 모델 OCR(Open Code Reasoning) 공개
엔비디아가 코드 생성과 문제 해결에 최적화된 추론 모델을 오픈 소스로 공개. 코딩 AI 분야 시장이 급성장하며 폐쇄형 모델의 대안을 제시했다는 점에서 주목.
라이브코드벤치(LiveCodeBench)에서 오픈AI의 'o3-미니'와 'o1-로우(low)' 능가. 오픈 소스 모델 중 코드 추론 성능에서 최상위권 기록. 
상업적 용도로 사용 가능 오픈.

- 엔비디아는 최근 고성능 코드 추론을 위한 ‘OCR(Open Code Reasoning)’ 제품군을 소개하고, 가중치와 데이터셋을 허깅페이스에 오픈 소스로 공개
- OCR-네모트론-32B(OpenCodeReasoning-Nemotron-32B) OCR-네모트론-14B OCR-네모트론-7B 등 세가지 규모로 제공되며, 모두 상업적 용도로 사용 가능.
- 코드 생성을 위한 추론을 위해 각각 '큐원2.5-32B-인스트럭트', '큐원2.5-14B-인스트럭트', '큐원2.5-7B-인스트럭트'를 미세조정했으며, 컨텍스트 창은 3만2000 토큰을 지원
- OCR 모델은 디버깅, 코드 생성, 논리 완성 등 실제 개발 환경에서 요구되는 복잡한 코드 추론 작업을 수행하도록 설계
- 코드 중심 벤치마크인 라이브코드벤치(LiveCodeBench)에서 오픈AI의 'o3-미니'와 'o1-로우(low)'를 능가하는 성능.
- 32B 모델은 오픈 소스 모델 중 코드 추론 성능에서 최상위권을 기록.
- 엔비디아에 따르면 이번 성능 향상의 핵심은 'OCR 데이터셋'이라는 고품질 코드 중심 학습 데이터를 바탕으로 학습.
- 이를 통해 명령어 수행, 다단계 문제 해결, 논리 추론 등에 최적화.
- 토큰 효율성이 최대 30% 향상, 적은 토큰으로도 더 정확하고 논리적인 코드 출력을 생성할 수 있다고 강조

huggingface https://huggingface.co/nvidia/OpenCodeReasoning-Nemotron-32B
huggingface: https://huggingface.co/datasets/nvidia/OpenCodeReasoning

## 중국 샤오미, 휴대폰용 추론 모델' 오픈 소스 출시
중국의 샤오미가 휴대폰에 탑재할 수 있는 오픈 소스 추론 모델을 출시. 
매개변수 70억개(7B)의 소형모델(sLM)로, 동급인 오픈AI의 'o1-미니'에 맞먹는 성능을 냈으며 '딥시크'와 '큐원' 등을 능가했다고 강조.
- 샤오미는 30일 깃허브를 통해 '미모-7B(MiMo-7B)' 제품군을 오픈 소스로 공개.
- 추론을 위한 강화 학습(RL)을 통해 훨씬 더 큰 32B 모델보다 뛰어난 성능을 보인다고 밝혀.
- 일반 지도 미세조정(SFT) 모델에 대한 RL을 통해 개발한 '미모-7B-RL'은 수학과 코드 추론 작업에서 o1-미니와 유사한 성능을 보인다고 주장
- 공개한 벤치마크에 따르면 '라이브코드벤치'와 'AIME 2024~2025'에서 o1-미니에는 10점 차로 뒤지지만,
- '딥시크-증류-7B-RL'는 물론 4배나 매개변수가 많은 '큐원 2.5-32B-RL-제로'의 성능을 넘어서
- 샤오미도 지난 2023년 자체 모델인 '미LM(MiLM)'을 개발해 휴대폰에 탑재해 왔고 이 때문에 이번 추론 모델도 작은 크기로 개발. 
- 이 가운데 수준급 오픈 소스 추론 모델을 내놓았다는 점으로 인해 본격적인 AI 경쟁에 합류한 것이라는 평. 
- 특히, 샤오미 모델은 sLM으로 온디바이스나 엣지 AI에 적합하다는 차별점.
- 
https://huggingface.co/XiaomiMiMo/MiMo-7B-Base
https://github.com/XiaomiMiMo/MiMo.git

## ReTool: Reinforcement Learning for Strategic Tool Use in LLMs (ByteDance, April 2025)
Paper: https://arxiv.org/abs/2504.11536 

“While reasoning models (e.g., DeepSeek R1) trained with reinforcement learning (RL), excel in textual reasoning, they struggle in scenarios requiring structured problem-solving, such as geometric reasoning, concise computation, or complex equation solving-areas where computational tools like code interpreters (CI) demonstrate distinct advantages. To bridge this gap, we propose ReTool, which enhances long-form reasoning with tool-integrated learning, including two key features: (1) dynamic interleaving of real-time code execution within natural language reasoning processes, and (2) an automated RL paradigm that allows policy rollouts with multi-turn real-time code execution and teaches the model in learning when and how to invoke tools based on outcome feedback. ReTool employs a systematic training framework, beginning with synthetic cold-start data generation to produce code-augmented long-form reasoning traces for fine-tuning base models. Subsequent RL training leverages task outcomes as rewards to iteratively refine the model's tool use strategy, enabling autonomous discovery of optimal tool invocation patterns without human priors. Experiments on the challenging MATH Olympiad benchmark AIME demonstrate ReTool's superiority: Our 32B model achieves 67% accuracy with 400 training steps, outperforming text-based RL baseline (40% accuracy, 1080 steps) in efficiency and performance. Remarkably, ReTool-32B attains 72.5% accuracy in extended settings, surpassing OpenAI's o1-preview by 27.9%. Further analysis reveals emergent behaviors such as code self-correction, signaling an ''aha moment'' in which the model autonomously masters adaptive tool use. These findings highlight the promise of outcome-driven tool integration for advancing complex mathematical reasoning and offer new insights into hybrid neuro-symbolic systems.”
Project Page: https://retool-rl.github.io/

## *RepText: 오픈소스 T2I 모델로 비공개 다국어 모델급 텍스트 렌더링 달성
 사전 훈련된 단일 언어 텍스트-이미지(T2I) 모델, 특히 최신 DiT 기반 모델이 다국어 시각적 텍스트를 정확하게 렌더링(복제)하도록 하는 RepText 프레임워크를 제안합니다. 텍스트 이해가 렌더링의 필수 조건은 아니라는 가정 하에, RepText는 ControlNet 방식에 원하는 텍스트의 캐니 엣지와 위치 이미지를 조건으로 사용하고 텍스트 인식 손실(OCR 기반)을 추가하여 학습합니다. 추론 시에는 '글리프 잠재 공간 복제' 기법으로 초기화하고 '영역 마스킹'을 적용하여 텍스트 정확도와 이미지 품질을 향상시킵니다.실험 결과, RepText는 기존 오픈소스 방법보다 우수하며 비공개 다국어 모델과 유사한 성능을 보입니다.
https://huggingface.co/papers/2504.19724

## 스마트폰을 자동화하는 LLM 에이전트: 프레임워크, 모델링, 데이터, 과제 총정리
 LLM(거대 언어 모델) 기반 스마트폰 GUI 에이전트 자동화 기술의 발전과 전망을 체계적으로 검토합니다.기존 스크립트 기반 자동화의 한계를 LLM의 발전된 언어 이해, 멀티모달 인식, 의사결정 능력으로 어떻게 해결하는지 설명합니다. 에이전트 프레임워크(단일/다중 에이전트, 계획-후-실행), 모델링 접근법(프롬프트 엔지니어링, 학습 기반), 데이터셋 및 벤치마크를 포함한 분류 체계를 제안합니다. 또한, 작업별 아키텍처, 지도 미세 조정(SFT), 강화 학습(RL) 전략 등 기술적 세부 사항과 함께 데이터셋 다양성, 기기 내 배포 효율성, 보안 등 향후 과제를 논하며, 연구자와 실무자를 위한 참조 자료를 제공합니다.
https://huggingface.co/papers/2504.19838

## 현실적인 대화 생성을 위한 Dia: 1.6B 파라미터 TTS 모델 및 오픈소스 공개

Nari Labs에서 개발한 1.6B 파라미터 텍스트-음성 변환(TTS) 모델인 Dia를 소개합니다. 
Dia는 텍스트 스크립트에서 직접적으로 매우 현실적인 대화 생성을 목표로 합니다. 기존 TTS 모델의 단조로운 음성 생성을 넘어서, Dia는 오디오 프롬프트를 통한 감정 및 어조 제어, 그리고 웃음, 기침 등 비언어적 표현 생성을 가능하게 하는 기능을 포함합니다. 연구 가속화를 위해 사전 훈련된 모델 체크포인트와 추론 코드를 Hugging Face를 통해 공개적으로 제공합니다(현재 영어만 지원). 또한, 모델의 기능을 시연하고 비교할 수 있는 데모 페이지와 ZeroGPU Space 환경을 제공하여 접근성을 높였습니다. Dia는 화자 태그([S1], [S2])를 이용한 대화 생성, 비언어적 표현 합성, 오디오 프롬프트를 이용한 음성 복제 등 다양한 기능을 지원하며, 연구 및 개발 커뮤니티의 기여를 장려하기 위해 Apache 2.0 라이선스로 배포됩니다. Dia는 고품질 GPU 환경에서 실시간에 가까운 추론 속도를 보이며, 향후 최적화 및 양자화 버전 추가를 계획하고 있습니다.

https://github.com/nari-labs/dia
## 네이버, 상업용 오픈소스 모델 HyperCLOVA X SEED 공개
네이버가 한국의 소버린 AI(Sovereign AI) 생태계 조성을 목표로 공개한 상업적 활용 가능 오픈소스 AI 모델군, HyperCLOVA X SEED를 공개했습니다. 이는 단순 모델 배포를 넘어, 기업 및 개발자가 자체 AI 역량을 강화하고 특정 비즈니스 요구사항에 맞춰 모델을 튜닝하여 활용할 수 있도록 지원하기 위함입니다. HyperCLOVA X SEED는 3B, 1.5B, 0.5B 파라미터 크기의 세 가지 모델로 구성되어 있으며, 각기 다른 강점을 지닙니다. 특히 3B 모델은 한국어 및 문화 맥락에 특화된 이미지 이해(vision understanding) 능력을 갖추고 있으며, 1.5B 모델은 지시 이행(instruction following) 능력, 0.5B 모델은 경량 환경에서의 자연스러운 한국어 대화 능력을 특징으로 합니다. 이 모델들은 Hugging Face를 통해 배포되며, 한국어 관련 벤치마크에서 동급 크기의 경쟁 모델 대비 우수한 성능을 보였으며, 특히 0.5B 모델은 높은 학습 비용 효율성을 달성했습니다. HyperCLOVA X SEED 공개는 국내 AI 기술 혁신과 생태계 확장에 기여할 것으로 기대됩니다.
https://tinyurl.com/4zbfsdp5

## 엔비디아, 최신 동영상 생성 기술 TTT(Test-Time Training)로 AI 애니메이션 ‘톰과 제리’ 제작 

엔비디아가 1분 길이의 복잡한 이야기를 일관된 스타일로 풀어내는 동영상 생성 인공지능(AI) 기술을 선보여. 이를 통해 생성한 ‘톰과 제리’는 놀라운 재현율을 보여줌.
-엔비디아와 스탠포드대학교 연구진은 13일 트랜스포머 아키텍처를 활용해 1분 분량의 멀티 샷 동영상을 일관성 있게 생성할 수 있는 새로운 기법 ‘테스트-타임 훈련(Test-Time Training, TTT)’을 소개. 
-텍스트를 기반으로 한 동영상 생성 기술은 빠르게 발전하고 있지만, 여전히 긴 이야기 구조를 담아내는 데에는 한계. 오픈AI의 ‘Sora’, 구글의 ‘Veo’ 등 최신 확산 모델은 짧은 고화질 영상 제작에는 성공했지만, 대부분의 클립은 20초를 넘기지 못함. 또 문제는 단순한 영상 길이가 아니라, 스토리 전개와 장면 간의 흐름을 얼마나 일관성 있게 유지하느냐는 점.
-연구의 핵심은 바로 숨겨진 상태를 작고 유연한 신경망으로 구성한 ‘TTT 레이어’다. 이 레이어는 영상이 생성되는 추론 과정 중에도 계속 스스로 학습(self-supervised learning)하며 맥락에 적응해 나감. 이를 통해 캐릭터의 행동, 장면 간 연결, 이야기의 흐름을 실시간으로 파악하며 일관된 이야기 구조를 유지할 수 있게 된
-연구진은 이 TTT 레이어를 기존에 사전 학습된 트랜스포머 모델에 통합했고, 그 결과 텍스트로 구성된 스토리보드를 바탕으로 최대 1분 길이의 애니메이션 영상 생성에 성공
-이번 실험은 고전 애니메이션 ‘톰과 제리’ 시리즈를 바탕으로 큐레이션한 데이터셋을 활용해 진행. 100개 영상에 대한 인간 평가에서 평균 34 포인트 높은 점수를 기록하며 기존 기법들을 크게 앞서.
-연구에서 생성된 AI 버전의 ‘톰과 제리’ 영상은 SNS와 유튜브 등 인터넷상에서 큰 반응. "놀라운 진전", "원작을 현대적으로 재해석했다"고 평가
-반면, AI가 예술을 훼손하고 있다는 반응도. “원작이 더 낫다” “별로 웃기지도 않는다. 오리지널이 최고” “진짜 애니메이터들이 무덤 속에서 뒤척일 듯” “기술은 좋지만 장인 정신이 사라지고 있다”
One-Minute Video Generation with Test-Time Training
https://test-time-training.github.io/video-dit/
https://github.com/test-time-training/ttt-video-dit

## 인간 동작 생성·편집하는 AI 모델 
베이징대학교 논문 및 코드. 
- project page : https://awfuact.github.io/motionrefit/
- paper : https://arxiv.org/abs/2503.20724
- code : https://github.com/emptybulebox1/motionRefit/

사실적인 인간 동작을 자동으로 생성하고 수정할 수 있는 새로운 인공지능(AI) 모델이 개발. 이 기술은 애니메이션 제작자, 게임 개발자, 영상 콘텐츠 제작자에게 유용한 도구로 활용될 전망

- 베이징대학교 연구진은 최근 인간 캐릭터 혹은 아바타의 사실적인 움직임을 간편하게 생성할 수 있는 새로운 AI 모델을 개발, 'CVPR 2025'에서 논문을 발표. 논문 제목 'Dynamic Motion Blending for Versatile Motion Editing)'
- 이들이 제안한 인간 동작 생성 방식은 MotionCutMix라는 데이터 증강 기법과 MotionReFit이라는 확산 모델을 기반. 지금까지 인간의 동작을 처음부터 생성하는 기술은 많은 발전을 이뤘지만, 이미 존재하는 동작을 자연스럽게 편집하는 기술은 드물었음.
- 모션컷믹스는 이러한 목표를 달성하기 위해 고안된 학습 기법으로, 3D 인간 동작을 텍스트 지시를 기반으로 학습하고 편집할 수 있도록 AI를 훈련. 단순히 ‘어떤 동작을 하느냐’뿐만 아니라 ‘어떻게 하느냐’와 같은 스타일 요소까지 편집할 수 있어 훨씬 정교한 창작이 가능. 소수의 주석 예시만으로도 수백만개의 훈련 변형을 생성할 수 있으며, 학습 속도에도 큰 영향을 주지 않음. 즉, 모션컷믹스는 다양한 학습 샘플을 생성하는 데이터 증강 기법.
- 모션리핏은 인간 동작을 실제로 생성하고 수정하는 오토리그레시브(autogressive) 확산 모델. 사용자가 텍스트로 원하는 동작 변화를 입력하면, 지시에 따라 인간 동작 시퀀스를 정밀하게 편집. 특정 부위에만 변화를 주는 공간적 편집(spatial editing)과 동작의 흐름이나 시간상 변화를 반영하는 시간적 편집(temporal editing)을 모두 지원하면서도 사용자로부터 부가적인 정보나 명확한 신체 부위 지정 없이 작동.
- "모션컷믹스는 고질적인 편집 데이터 부족 문제를 극복하고, 기존 동작 데이터를 활용해 사실상 무한에 가까운 학습 데이터를 생성할 수 있는 효과적인 해법이라는 점을 입증"
- 사람이나 휴머노이드 캐릭터가 포함된 다양한 콘텐츠의 생성과 편집에 활용될 수 있으며, 특히 애니메이션 제작자, 게임 개발자, 영상 콘텐츠 제작자에게 유용한 도구 기대. 또 텍스트 기반 인터페이스를 사용하기 때문에, 게임이나 애니메이션 제작 경험이 없는 일반 사용자도 사용 가능. 로봇공학 분야로 확장, 서비스 로봇의 움직임 개선에도 응용될 수 있을 것.

 
## NPU인 Ascend 칩을 활용해서 학습한 모델인 Pangu Ultra를 공개
논문제목은 Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs
arXiv: https://arxiv.org/abs/2504.07866

크기는 135B 매개변수 MoE구조가 아닌 싱글 트랜스포머 94층 모델
13.2조개 토큰으로 프리트레이닝하고 이후 리즈닝 능력을 위한 포스트 트레이닝
컨텍스트 길이도 4k 에서 128k로 확장
8192 Ascend 칩 (아마도 910b로 추정) 를 각 노드(서버)별 8장을 HCCS(Huawei Cache Coherence System - 엔비디아로 치면 NVLink에 해당) 로 연결하고 각 노드는 RoCE 200 Gbps (인피니밴드에 해당) 로 연결
병렬 학습인 데이터, 텐서, 시퀀스, 파이프라인 4가지 병렬 학습 기법을 모두 적용
MFU (실제 연산 활용율) 50% 정도를 달성
910b 제원이 64GB 메모리에 속도가 1.6 TiB/s 정도이므로 A100 사양 정도. 

## ACTalker (Tencent): Audio-visual driven Talking Head Video Generation
중국 Tencent에서 발표한 논문으로 오디오 기반 또는 얼굴 모션 기반 등 다양한 신호로 구동되는 자연스러운 얼굴 영상 생성 가능. 기존의 Sadtalker, Hallo, Ecomimic, EDTalk, Memo 등보다 우수 주장
ACTalker, Audio-visual Controlled Video Diffusion with Masked Selective State Spaces Modeling for Natural Talking Head 

Generation (2504, HKUST,Tencent,Tsinghua University)

[project: https://harlanhong.github.io/publicat/actalker/index.html
](https://harlanhong.github.io/publications/actalker/index.html?fbclid=IwY2xjawJjq91leHRuA2FlbQIxMAABHi9Frpk3LYGL9QObKoZYeS_ZR19X6V7rfm2waMPDnpaXLW9PO9hdravB3LdQ_aem_8EY9pZUGfXKpdjoRlLQPOg)

paper : https://arxiv.org/abs/2504.02542

## Inference-Time Scaling for Generalist Reward Modeling
o1이나 R1과 같은 Large Reasoning Model은 Inference Time Scaling 즉 학습이 아닌 인퍼런스 할때 더 많은 연산을 해서 더 많은 reasoning token을 생성하고  더 길게 추론적 사고(리즈닝) 를 할수록 더 정확한 결과를 만들어 낸다는 경험적 법칙이지요. 이 과정에서 핵심은 Reward model을 정확하게 만들어놔야 가능합니다. 
현존 리즈닝 모델들이 과학, 수학, 코딩 등에서 강력한 Inference Time Scaling 을 보여주고 있는데 이 데이터들은 질문과 풀이과정이나 규칙들이 명확해서 Reward model을 만드는 것이 쉬운편이었는데 다양한 일상 대화에서 insturction following이나 복잡한 대화 같은 것들은 보상 점수 평가가 애매한 부분이 많습니다. 이번 GRM은 Generalist Reward Modeling 라는 이름에서 보듯이 이런 부분을 해결하기 위한 방법으로 제안된 것 같네요.
이 논문에서 새롭게 제안한 것이 Self-Principled Critique Tuning (SPCT)입니다. 즉 강화학습의 Reward model 학습에 셀프로 Principle 원칙과 Critique (리워드를 단순 숫자값이 아닌 이유를 말로 설명하는 것)을 생성하고 이걸 기반으로 학습니다. 특성으로 치면 Critique를 리워드로 생성하는 Generative 리워드기법과 동일 질의에 두가지 답을 함께 넣고 비교를 기반으로 하는 Point-wise 리워드기법을 합한 개념으로 볼수 있네요 (그림 참조) 
절차는 불량한 생성 데이터를 빼면서 튜닝하는 rejective fine-tuning 으로 cold start를 하고 GRPO를 활용해서 RL을 수행합니다. Inferece Time Scaling 효과를 위해 두가지를 제안하는데 하나는 여러 결과를 단순 Voting, 다른 하나는 Principle과 Critique를 평가하는 pointwise RM인 MetaRM을 별도로 두고 Voting을 가이드하는 기법입니다.
얘네들을 다합한 DeepSeek-GRM 은 다양한 RM Bench를 갖고 평가를 하는데 기존 단순 스칼라에비해 더많은 리즈닝 후보를 만들어 낼수록 점점 평가점수가 개선되는 것을 확인할 수 있습니다. 특히 DeepSeek-GRM에 사용된 모델이 Gemma-2-27B 모델이라 상당히 컴팩트한 모델인데요. 이 가벼운 모델로도 큰 모델들보다 더 고품질 혹은 거의 유사한 품질을 만들어 내네요. MetaRM을 쓰는 경우가 성능이 가장 좋은데 MetaRM은 별도로 Gemma2-27B로 만들었다 하니.. 모델 크기가 2배가 되는 단점이 ㅎㅎㅎ
이 논문의 장점은 기존 LRM들의 Reward Modeling 을 잘 비교분석해두었고 또 디테일한 정보들이 많이 적혀있어 연구자들의 연구에도 매우 유용할 뿐아니라 다른 기업이나 연구그룹이 바로 시도해 볼 수 있다는 점입니다. 그리고 이 기법으로 이제 리즈닝 모델이 수학, 과학, 코딩을 넘어 일상의 에이전트로 확산이 가속화 될 수 있겠습니다. 
더불어 딥시크를 엔지니어링이나 가성비로만 평가하시는 분들이 계신데 연구역량도 매우 뛰어난 그룹임을 다시한번 증명한 케이스네요

논문: https://arxiv.org/abs/2504.02495

## AKD (NVIDIA): Text to skeleton-based Character Animation
NVIDIA에서 발표한 스켈레톤 기반 애니메이션과 최신 제너레이티브 모델의 강점을 결합하여 고충실도 캐릭터 애니메이션을 생성하는 프레임워크 논문. 리깅된 3D 에셋에 스켈레톤 기반 표현을 사용하여 관절 수준 제어에 집중함으로써 자유도(DoF)를 대폭 줄여 효율적이고 일관된 모션 합성 가능. 실험 결과 우수한 3D 일관성과 모션 품질을 달성 주장. 
AKD: Articulated Kinematics Distillation from Video Diffusion Models (2504, UCLA, NVIDIA)
project: https://research.nvidia.com/labs/dir/akd/

paper : https://arxiv.org/abs/2504.01204

## Overtrained Language Models Are Harder to Fine-Tune"
- "더 오래 훈련할수록 더 좋다"는 LLM 상식이 항상 맞는 건 아님! 
- 사전 학습을 너무 많이 하면 오히려 미세 조정 성능이 떨어진다는 걸 발견
- 이걸 "재앙적 과다 훈련(catastrophic overtraining)"이라고
원인
- 모델이 오래 학습될수록 파라미터가 변화에 더 민감해짐. 작은 변경에도 큰 성능 저하가 발생하는 "점진적 민감도(progressive sensitivity)" 현상이 일어남.
해결 전략
- 낮은 러닝레이트 사용
- 충분한 정규화 활용
- 최신 체크포인트가 아닌 "최적 체크포인트" 사용
- data replay 같은 방법도 도움(이미 활용...)
- https://arxiv.org/pdf/2503.19206

## Mureka AI, New Music Production.
음악 제작이 완전히 새로워졌습니다. Mureka AI를 사용하면 누구나 사용자 지정 보컬, 나만의 사운드, 심지어 훈련된 모델을 사용하여 아이디어를 완벽한 트랙으로 만들 수 있습니다.
Mureka AI는 일반적인 음악 생성기와는 다릅니다. 사용자 지정 보컬과 간편한 컨트롤로 크리에이터가 고품질의 개인 맞춤형 트랙을 빠르게 만들 수 있습니다. 또한 최초의 생각의 연결(COT) 모델을 갖춘 플랫폼이므로 음악에 실제 논리가 있습니다, 
곡을 만드는 단계별 방법
- 참조 트랙으로 시작하거나 아이디어를 입력하세요
- 보컬 선택- 프리셋 사용 또는 직접 업로드
- 생성을 클릭하고 Mureka가 마법을 부리게 해주세요
- 완성된 트랙 다운로드 
Mureka와 Suno 비교
Suno가 대중적인 선택일 수 있지만 Mureka는 훨씬 더 많은 제어와 창의적인 유연성을 제공합니다. 
Mureka를 사용하면:
- 텍스트뿐만 아니라 오디오를 프롬프트로 사용
- 트랙 전체에서 보컬을 일관되게 유지
- 나만의 맞춤형 모델 훈련
- 오픈 API에 액세스하여 다른 도구와 연결하거나 구축할 수 있습니다. 
Mureka_AI를 돋보이게 하는 또 하나의 이유는, 유일하게 COT(Chain of Thoughts)라는 것을 사용한다는 점입니다.
실제 작곡가가 생각하는 것과 같은 구조와 흐름을 가진 음악을 인공지능이 만들 수 있도록 도와주는 모델입니다.

## 미스트랄 AI, '젬마 3' 성능 넘는 오픈 소스 sLM 'Mistral Small 3.1' 공개
-미스트랄 AI는 17일 ‘미스트랄 스몰 3.1(Mistral Small 3.1)'이라는 새로운 소형 LMM을 무료로 공개.
-텍스트와 이미지를 동시에 처리할 수 있고, 최대 12만8000 토큰의 컨텍스트 창을 제공. 초당 150 토큰의 처리 속도로 빠른 응답을 요구하는 애플리케이션에 적합.
-단일 'RTX 4090'이나 32GB RAM이 탑재된 맥(Mac)에서 실행할 수 있다는 것이 특징.
-벤치마크에서도 뛰어난 결과를 기록했다. 텍스트 지시와 멀티모달 지시, 다국어 지원, 긴 문맥 처리 등 다양한 테스트에서 젬마 3와 'GPT-4o 미니' 등보다 우수한 성능.
-미스트랄 스몰 3.1은 허깅페이스에서 다운로드 가능. 미스트랄 AI의 개발자 플랫폼인 ‘La Plateforme’과 구글 클라우드 Vertex AI에서도 바로 사용 가능.

## ChatGPT부터 DeepSeek-R1까지 주요 LLM의 Post-training 기법을 포괄적으로 정리한 논문
https://arxiv.org/abs/2503.06072

## Google Open LLM Gemma 3 Release! 
Gemma 3 소개: 단일 GPU 또는 TPU에서 실행할 수 있는 최고 성능의 모델 (Google Blog, Mar 12, 2025)
Gemma 개방형 모델 제품군은 유용한 AI 기술에 대한 접근성을 높이기 위한 우리의 노력의 토대입니다. 지난달에는 1억 건 이상의 다운로드라는 놀라운 채택과 6만 개 이상의 Gemma 변형을 만들어낸 활기찬 커뮤니티가 탄생한 Gemma의 첫 생일을 축하했습니다. 이러한 젬마버스는 계속해서 저희에게 영감을 주고 있습니다.
오늘은 Gemini 2.0 모델과 동일한 연구와 기술을 바탕으로 제작된 경량의 최첨단 개방형 모델 컬렉션인 Gemma 3를 소개합니다. 이 제품은 휴대성이 뛰어나고 책임감 있게 개발된 가장 진보된 개방형 모델입니다. 휴대폰, 노트북, 워크스테이션 등 다양한 기기에서 직접 빠르게 실행되도록 설계되어 개발자가 필요한 모든 곳에서 AI 애플리케이션을 개발할 수 있도록 지원합니다. Gemma 3는 다양한 크기(1B, 4B, 12B, 27B)로 제공되므로 특정 하드웨어 및 성능 요구 사항에 가장 적합한 모델을 선택할 수 있습니다.
개발자가 Gemma 3에서 사용할 수 있는 새로운 기능
-세계 최고의 단일 가속기 모델로 빌드: Gemma 3는 LMArena의 리더보드에서 실시한 사전 선호도 평가에서 Llama-405B, DeepSeek-V3, o3-mini보다 뛰어난 성능을 발휘하며 크기 대비 최첨단 성능을 제공합니다. 이를 통해 단일 GPU 또는 TPU 호스트에 적합한 매력적인 사용자 경험을 만들 수 있습니다.
-140개 언어로 글로벌 진출: 고객의 언어를 지원하는 애플리케이션을 빌드하세요. Gemma 3는 35개 이상의 언어에 대한 기본 지원과 140개 이상의 언어에 대한 사전 학습된 지원을 제공합니다.
-고급 텍스트 및 시각적 추론 기능을 갖춘 AI를 제작: 이미지, 텍스트, 짧은 동영상을 분석하는 애플리케이션을 쉽게 구축하여 대화형 지능형 애플리케이션의 새로운 가능성을 열어줍니다.
-확장된 컨텍스트 창으로 복잡한 작업을 처리: Gemma 3는 애플리케이션이 방대한 양의 정보를 처리하고 이해할 수 있도록 128k 토큰의 컨텍스트 창을 제공합니다.
-함수 호출을 사용하여 AI 기반 워크플로우를 생성: Gemma 3는 함수 호출과 구조화된 출력을 지원하여 작업을 자동화하고 에이전트 경험을 구축할 수 있도록 도와줍니다.
-정량화된 모델을 통해 더욱 빠른 고성능 제공: Gemma 3는 공식 정량화된 버전을 도입하여 모델 크기와 계산 요구 사항을 줄이면서도 높은 정확도를 유지합니다.

## Ai2, 'deepseek-V3' 능가하는 오픈 소스 405B LLM 'Tülu 3 405B' 공개..."사후 훈련에 중점" (출처: 뉴스)
-앨런 AI연구소(Ai2)가 강화 학습과 추론 강화를 포함한 사후 훈련(post-training)에 중점을 둔 새로운 오픈 소스 모델을 공개.
-역대 최대 규모의 오픈 소스 대형언어모델(LLM) 'deepseek-V3'를 능가한다고 강조.
-안전성 벤치마크에서 V3와 '라마 3.1', 누스의 '헤르메스 3'를 능가하는 결과. 10가지 AI 벤치마크 테스트에서는 80.7점을 기록, V3의 75.9점을 초과. 'GPT-4o'의 81.6점에는 미치지 못했지만, GPT-4o 및 V3와 경쟁할 수 있는 강력한 모델로 평가
-V3가 집중 조명을 받은 뒤 이를 능가한다는 모델 줄이어, 알리바바가 최신 모델 Qwen2.5-Max를 출시하며 역시 V3를 넘어섰다고 발표

## Mol-LLaMA: 분자 이해를 위한 새로운 다중 모드 AI 모델
Mol-LLaMA는 분자의 일반적인 이해를 위해 다중 모드 명령어 튜닝을 통해 분자 중심의 일반 지식을 파악하는 대규모 분자 언어 모델입니다. 분자의 기본 기능을 포함하고 분자 구조의 필수 지식을 통합하는 주요 데이터 유형을 설계했습니다. 또한 서로 다른 분자 표현의 고유한 이점을 활용하여 서로 다른 분자 인코더의 보완 정보를 통합하는 모듈을 도입하여 분자 특징에 대한 이해도를 높였습니다. 실험 결과 Mol-LLaMA는 분자의 일반적인 특징을 이해하고 사용자 쿼리에 대한 관련 응답을 자세한 설명과 함께 생성할 수 있는 것으로 나타났습니다. 이는 범용 분자 분석 지원 도구로서의 잠재력을 의미합니다.
https://huggingface.co/papers/2502.14776

## SurveyX: LLM 기반 자동 설문 조사 생성 시스템
SurveyX는 대규모 언어 모델(LLM)을 사용하여 학술 설문 조사를 자동으로 생성하는 시스템입니다. 이 시스템은 온라인 참조 검색, 속성 트리라는 사전 처리 방법, 재연마 프로세스를 도입하여 설문 조사 작성 효율성을 크게 높입니다. 실험 평가 결과 SURVEYX는 기존 자동 설문 조사 생성 시스템보다 콘텐츠 품질(0.259 향상)과 인용 품질(1.76 향상) 면에서 뛰어나 여러 평가 차원에서 인간 전문가의 성능에 근접하는 것으로 나타났습니다. SurveyX에서 생성한 설문 조사의 예는 프로젝트 웹사이트에서 확인할 수 있습니다.
https://huggingface.co/papers/2502.13449

## Step-Video-T2V(StepFun), 30B open-source text-to-video generation model(chinese)
중국 StepFun에서 공개한 오픈소스 30B 파라미터와 최대 204프레임 길이의 동영상을 생성할 수 있는 모델. 
영어와 중국어 지원. 
Hunyuan 오픈소스 및 Gen-3 등 상용모델보다 우수 주장. 
80GB VRAM 권장.  

Key Features:
- 30B parameters for high-quality Text-to-Video
- Supports both Chinese & English Prompts 
- Generates high-quality 540P videos (up to 204 frames) 
- Strong dynamics, consistency, and stunning
- Fully Open-Source: MIT licensed
- 
 30B 파라미터와 최대 204프레임 길이의 동영상을 생성할 수 있는 사전 학습된 최첨단 텍스트-비디오 사전 학습 모델인 Step-Video-T2V를 소개합니다. 딥 압축 가변 자동 인코더인 Video-VAE는 비디오 생성 작업을 위해 설계되어 16x16 공간 및 8x 시간 압축률을 달성하는 동시에 뛰어난 비디오 재구성 품질을 유지합니다. 사용자 프롬프트는 영어와 중국어를 모두 처리하기 위해 두 개의 이중 언어 텍스트 인코더를 사용하여 인코딩됩니다. 3D 풀 어텐션 DiT는 플로우 매칭을 사용하여 훈련되며 입력 노이즈를 잠재 프레임으로 노이즈 제거하기 위해 사용됩니다. 비디오 기반 DPO 접근 방식인 Video-DPO를 적용하여 아티팩트를 줄이고 생성된 비디오의 시각적 품질을 개선합니다. 또한 훈련 전략을 자세히 설명하고 주요 관찰 사항과 인사이트를 공유합니다. 새로운 비디오 생성 벤치마크인 Step-Video-T2V-Eval을 통해 Step-Video-T2V의 성능을 평가하여 오픈 소스 및 상용 엔진과 비교했을 때 최첨단 텍스트-비디오 품질을 입증합니다. 또한 현재 확산 기반 모델 패러다임의 한계에 대해 논의하고 비디오 기반 모델의 미래 방향에 대해 설명합니다. Step-Video-T2V와 Step-Video-T2V-Eval은 이 https URL에서 확인할 수 있습니다. 온라인 버전도 이 https URL에서 액세스할 수 있습니다. 저희의 목표는 비디오 재단 모델의 혁신을 가속화하고 비디오 콘텐츠 제작자의 역량을 강화하는 것입니다.
paper : https://arxiv.org/abs/2502.10248
huggingface : https://huggingface.co/stepfun-ai/stepvideo-t2v
demo(chinese) : https://yuewen.cn/videos

## SkyReels V1 
SkyReels V1 is the first and most advanced open-source human-centric video foundation model. 
By fine-tuning HunyuanVideo on O(10M) high-quality film and television clips

## Open R1 Update #2
DeepSeek-R1-Distill 모델을 재현하기 위한 Hugging Face의 노력의 두 번째 업데이트가 공개되었습니다. 
이번 업데이트에서는 Numina Math 1.5 데이터셋 기반, DeepSeek-R1을 통해 reasoning trace를 생성한 OpenR1-Math-220k 데이터셋을 공개한 것이 주 내용입니다.
아 내가 DeepSeek-R1으로 DeepSeek-R1과 동일한 방법을 통해, Domain Specific 문제를 풀어보고 싶다는 분들께서 참조해볼 만한 사항들을 읽어볼 수 있습니다.
특히 합성 reasoning trace 데이터셋을 생성하는 부분인데, vLLM 및 SGLang을 활용해서 데이터를 생성하는 경우 H100 한 장에 최대 25개의 reasoning trace 생성이 가능하다고 합니다 (정확히는 H100 한 장에 DeepSeek-R1이 들어갈 수 없고, 최소 8 x H100이 필요하므로, 한 장에 대한 추정치 같습니다). 
: 약 800k 건의 trace 생성에 512 x H100을 사용하여, 약 2일 정도 작업 된 것 같습니다.
: 원본 400k에 대해 최소 두 개 이상의 solution을 생성해서 800k를 구축한 사례입니다 (생성된 답이 옳을 수도 있고, 아닐 수도 있어서 추후 필터링을 통해 올바른 놈들만 속아내기 위함)

그리고 800k 생성된 데이터로부터 Rule 및 Llama-3.3-70B-Instruct 모델 기반 평가를 통해 최종적으로 225k 정도의 합성 데이터를 생성했고, 이를 오픈 소스로 공개하였습니다. 여기에는 단일 문제에 대해 여러 건의 옳은 답이 포함된 경우도 있고, 단일 문제 - 단일 답만 포함된 경우가 존재합니다.
: 전자는 DPO를 통해 추가 preference 학습에 활용해 봤다고 하는데, 결론적으로 도움이 되지는 않았다고 합니다.
결론적으로 이렇게 생성한 데이터로 Qwen-7B 모델을 학습시켰고, DeepSeek에서 공개했던 Distill-Qwen-7B 모델의 성능을 MATH-500, AIME24에 대해 어느정도 재현하였습니다 (약간은 못 미치는 수준)
어쨋든 이 과정의 결과물인 DeepSeek-R1을 GPU 멀티 클러스터로 서빙하여 합성 데이터셋을 생성하는 스크립트, 이를 학습시키는 스크립트, 결과를 빠르게 평가하기 위한 lighteval이 모두 오픈소스로 공개되어 있습니다. 관심 있는 분들은 살펴보는게 좋겠네요.
https://huggingface.co/blog/open-r1/update-2

## Janus-Pro Unified Multimodal Understanding and Generation with Data and Model Scaling

DeepSeek에서 최근 공개한 Janus-Pro에 대해 간단히 리뷰해 보았습니다. 그다지 팔로업을 많이 하지 않아서, 신선했습니다 ㅎㅎ. 알고보니 Janus (야누스) 라고 Janus-Pro 이전 버전의 모델이 존재했고, 이전 버전 모델 구조를 그대로 가져가되, 학습 방법론을 약간 변경하면 훨씬 더 좋은 성능에 도달 할 수 있었다는 것이 골자입니다.
Janus 에서 제시되었던 기본 학습 전략은 세 단계로 나뉘어지는 데, 이렇게 단계를 나누는 이유는 시각 정보를 이해하고 생성하기 위해 분리한 "인코더" 및 "디코더/헤드"를 조심스럽게 기초적인 것부터 복잡한 것까지 순차적으로 학습 및 실험하기 위해서 인 것 같습니다.
: 한 마디로.. 멀티모달 데이터의 이해 및 생성을 동시에 해내는 모델을 어떻게 학습시켜야 하는가에 대한 "감"을 잡기 위한 선행 연구 정도로 보입니다.
Janus Pro에서는 이전 연구를 통해 발견된 사실에 기반하여 각 세 단계 전략을 약간씩 손을 봤고, 모델의 아키텍쳐 자체에는 아무런 변화가 없습니다. 대충 이해하기로는 Stage1 / Stage2와 같은 기초 공사를 위한 단계에서는 데이터를 믹스하는 것보다는 목적별 데이터를 분리하여 학습을 진행하는 것이 좋아 보이며, Stage3의 SFT 단계에서도 데이터를 어떻게 잘 섞는지가 성능 결과에 영향을 미치기 때문에 멀티모달 능력을 끌어올리려면 해당 부분에 대한 데이터 비율을 다른 것에 맞춰 조정하면 좋다는 것입니다 (또는 다른 것들의 비율을 줄이거나)
벤치마크 데이터셋을 통해 본 성능은 TokenFlow XL, MetaMorph, SD3 Medium, DALL-E 3 등 멀티모달 입/출력을 지원하지 않는 전문화된(?) 다른 모델 및, 멀티모달 입/출력을 지원하는 다른 모델과 비교해서도 크게 성능이 떨어지지는 않습니다.
--------
 384x384 해상도의 이미지 입/출력만을 지원한다는 것입니다. 
 
## Comfy3D Update: (v0.1.4.alpha) Integrated Hunyuan3D-2
- Integrated Hunyuan3D-2
- (I did bunch of work including rewrote its mesh processor from C++ to Python to make sure only one additional custom package is required)
- Support newest Comfy UI v0.3.12, py12, cu124, torch 2.5.1, Windows 10/11 (Pre-built wheels are available)

github: https://github.com/MrForExample/ComfyUI-3D-Pack

## 딥시크가 오픈AI의 'o1' 모델과 경쟁하는 추론 모델 ‘DeepSeek R1’ 오픈 소스로 공개 
- 'V3' 모델로 세계 최고의 오픈 소스 모델이라는 평가를 받은 딥시크가 이번에는 오픈AI의 'o1' 모델과 경쟁하는 추론 모델 ‘R1’ 시리즈를 오픈 소스로 공개했다.
- 딥시크는 20일 오픈 소스 추론 모델인 ▲R1 ▲R1-제로(R1-Zero) ▲R1-증류(R1-Distill) 등을 공식 출시했다고 발표했다.
- R1과 R1-제로는 ‘딥시크-V3’를 미세조정한 모델로, 각각 6710억개의 매개변수를 포함하고 있다. 이 모델은 '전문가 혼합(MoE)' 아키텍처를 채택, 전체 매개변수 중 약 340억개만 활성화하도록 설계됐다. 즉, 추론 비용과 메모리 사용량을 줄이면서도 높은 성능을 유지한다.
- 추론 특화 LLM은 일반적으로 강화 학습(RL)과 지도 미세조정(SFT) 두가지 방법으로 학습된다. RL은 시행착오를 통해 AI가 작업을 수행하도록 훈련하는 방식이며, SFT은 작업 예시를 제공해 출력 품질을 향상하는 방식이다.
- 하지만, 딥시크는 R1-제로를 개발하는 과정에서 SFT를 생략했음에도 불구하고, 복잡한 작업을 단순한 하위 단계로 분해하는 등 주요 추론 기술을 성공적으로 구현했다. R1-제로는 추론 벤치마크(AIME 2024)에서 o1과 비슷한 성능을 기록했다.
- 딥시크는 "LLM의 추론 능력이 SFT 없이 RL로 유도될 수 있음을 증명한 최초의 공개 연구"라고 설명했다.
- 다만, R1-제로는 출력 품질에 한계가 있었다. 응답의 반복, 낮은 가독성, 언어 혼합 문제 등을 보이는 경우가 있었다. 이를 보완하기 위해 딥시크는 R1 모델을 개발했다.
- R1은 R1-제로의 개선 버전으로, 수정된 훈련 워크플로우를 적용했다. 여기에는 R1-제로 개발 시 생략했던 SFT가 포함됐다. 딥시크는 이를 통해 출력 품질을 크게 향상했다고 밝혔다.
- 벤치마크 결과, R1은 여러 분야에서 o1 모델을 능가했으며, o1이 더 높은 점수를 받은 경우에도 R1은 차이가 5% 내에 불과했다.
R1은 높은 성능뿐만 아니라 딥시크의 API를 통해 제공되며, 특히 비용은 o1 대비 90~95% 저렴하다는 강점이 있다.
또 딥시크는 하드웨어 효율성이 뛰어나지만 성능은 낮은 'R1-증류(Distillation)' 모델군도 오픈 소스로 공개했다. 여기에는 ▲R1-증류-큐원-1.5B ▲R1-증류-큐원-7B ▲R1-증류-라마-8B ▲R1-증류-큐원-14B ▲R1-증류-큐원-32B ▲R1-증류-라마-70B 등이 포함된다.
이 모델들은 R1에서 증류한 데이터를 기반으로, 메타의 '라마'와 알리바바의 '큐원'을 미세조정해 개발됐다. 특히, R1-증류-큐원-1.5B는 노트북에서도 실행 가능하며, R1-증류-큐원-32B는 여러 벤치마크에서 오픈AI의 o1-미니를 능가하는 성능을 보였다.
현재 R1 시리즈는 허깅페이스에서 모델 가중치와 코드를 다운로드하거나 API를 사용할 수 있으며, 딥시크 채팅 플랫폼을 통해 테스트해볼 수 있다.

모델 다운로드 : https://huggingface.co/deepseek-ai/DeepSeek-R1

##  MiniMax, 역대 최대 컨텍스트창 갖춘 오픈 소스 모델 MiniMax-01 공개..."AI 에이전트에 특화"
대형언어모델(LLM) ‘MiniMax-text-01’ , 비전언어모델(VLM)인 ‘MiniMax-VL-01’ 오픈소스 공개
최대 400만 토큰의 컨텍스트 창을 처리, 기존의 최고 수준인 '제미나이 1.5 프로'의 200만 토큰 컨텍스트 창보다 두배 더 확장된 용량
에이전트는 확장된 컨텍스트 처리 기능과 지속적인 메모리를 점점 더 필요
미니맥스-텍스트-01은 MMLU에서 88.5%의 정확도를 기록하며 GPT-4와 경쟁할 만한 성능 발휘, 미니맥스-VL-01은 DocVQA에서 96.4%, AI2D 벤치마크에서 91.7%의 정확도를 기록하며 다른 경쟁 모델들보다 우위.
huggingface : https://huggingface.co/MiniMaxAI
## '큐원' 미세조정한 추론 모델 오픈 소스 Sky-T1-32B-Preview 공개 
-합성 데이터를 활용한 훈련으로 비용 절감 
-'QwQ-32B-프리뷰'로 초기 데이터를 생성한 뒤 데이터를 선별하고, 오픈AI의 'GPT-4o-미니'로 실용적인 형식으로 재구성. 이 데이터로 추론 기능이 없는 ‘큐원2.5-32B-인스트럭트’를 미세조정
- 'H100' GPU 8대로 구성된 랙에서 약 19시간 동안 훈련, 비용 450달러 
- 수학 문제 모음 'MATH500'에서 초기 'o1-프리뷰' 보다 우수한 성능, 코딩 평가 데이터셋 '라이브코드벤치'의 난이도 높은 문제에서도 o1-프리뷰를 뛰어넘는 결과. 그러나 물리학과 생물학, 화학 등 관련 박사 수준의 문제를 다룬 'GPQA-다이아몬드'에서는 o1-프리뷰에 미치지 못함.
- 허깅페이스와 깃허브에 모두 공개
450달러(약 66만원)에 불과한 비용으로 훈련한 오픈 소스 인공지능(AI) 추론 모델이 등장했다. 파운데이션 모델을 개발한 것이 아니라 기존 모델로 합성 데이터를 만들고 이를 미세조정하는 등 '복제'에 불과하다고 밝혔지만, 최소한의 컴퓨팅 자원으로 경제적이고도 강력한 AI 시스템을 구축했다는 점에서 관심이 모인다.
UC 버클리의 연구실인 노바스카이(NovaSky)는 10일(현지시간) 깃허브를 통해 고급 추론 기능을 갖춘 오픈 소스 모델 ‘스카이-T1-32B-프리뷰(Sky-T1-32B-Preview)’를 공개했다. 
연구진은 “이 모델은 450달러 미만의 비용으로 훈련했으며, 효율적이면서도 저렴한 방식으로 고급 추론 능력을 구현할 수 있음을 입증했다”라고 밝혔다.
스카이-T1-32B-프리뷰는 합성 데이터를 활용한 훈련으로 비용을 절감했다. 연구진은 알리바바의 추론 모델 'QwQ-32B-프리뷰'로 초기 데이터를 생성한 뒤 데이터를 선별하고, 오픈AI의 'GPT-4o-미니'로 데이터를 실용적인 형식으로 재구성했다.
그다음, 이 데이터로 추론 기능이 없는 ‘큐원2.5-32B-인스트럭트’를 미세조정했다.
그 결과 탄생한 320개의 매개변수를 가진 스카이-T1-32B-프리뷰 모델은 엔비디아 'H100' GPU 8대로 구성된 랙에서 약 19시간 동안 훈련했다. 이 과정에 든 비용이 450달러에 못 미쳤다.
하지만 수학 문제 모음 'MATH500'에서 초기 'o1-프리뷰' 버전보다 우수한 성능을 보였으며, 코딩 평가 데이터셋 '라이브코드벤치'의 난이도 높은 문제에서도 o1-프리뷰를 뛰어넘는 결과를 나타냈다. 그러나 물리학과 생물학, 화학 등 관련 박사 수준의 문제를 다룬 'GPQA-다이아몬드'에서는 o1-프리뷰에 미치지 못했다.
연구진은 모델 훈련에 사용된 데이터셋과 필요한 훈련 코드를 허깅페이스와 깃허브에 모두 공개하여 누구나 이를 활용할 수 있도록 했다.
연구진은 "이번 연구는 높은 수준의 추론 기능을 저렴하고 효율적으로 복제할 수 있음을 보여준다"라고 강조했다.
또 "앞으로는 강력한 추론 성능을 유지하는 더 효율적인 모델을 개발하고 테스트 시 모델의 효율성과 정확성을 더욱 향상하는 고급 기술을 탐구하는 데 집중할 것"이라고 밝혔다.
이처럼 최근에는 적은 컴퓨팅 자원을 활용해 저비용 고품질의 AI 시스템을 구현하려는 흐름이 이어지고 있다. AI 기업 라이터가 발표한 ‘팔미라 X 004(Palmyra X 004)’ 모델은 약 70만달러(약 10억원)의 개발 비용으로 주목을 받았으며, 중국의 딥시크는 약 550만달러(약 80억원)의 훈련 비용으로 강력한 오픈 소스 추론 모델 ‘딥시크-V3(DeepSeek-V3)’를 개발해 화제가 됐다.
huggingface : https://huggingface.co/NovaSky-AI
github : https://github.com/NovaSky-AI/SkyThought
## rStar-Math: 작은 언어 모델의 수학적 추론 능력 향상을 위한 자체 진화형 딥 씽킹

1.5B~7B에 불과한 작은 모델로 OpenAI의 o1에 필적하는 수학적 추론 능력을 달성했답니다.
핵심 내용:
1. 주요 성과:
- 작은 규모(1.5B-7B)의 언어 모델로도 OpenAI의 o1과 비슷하거나 더 나은 수학 추론 능력 달성
- 상위 모델로부터의 지식 증류 없이 자체적인 진화 방식 사용
- MATH 벤치마크에서 Qwen2.5-Math-7B를 58.8%에서 90.0%로 향상
2. 주요 혁신:
- 코드 기반 CoT(Chain of Thought) 데이터 합성 방법 도입
- 새로운 프로세스 보상 모델 학습 방법 개발
- 4단계 자체 진화 레시피 구현
3. 핵심 방법론:
- Monte Carlo Tree Search(MCTS) 활용한 딥 씽킹 구현
- 단계별 검증된 추론 궤적 생성
- 프로세스 선호도 모델(PPM) 도입
4. 주요 발견:
- 내재적 자기 반성 능력의 출현
- PPM이 정리 적용 단계를 효과적으로 식별
- 보상 모델이 System 2 추론의 성능 상한을 결정
5. 실험 결과:
- MATH, AIME 2024, AMC 2023 등 다양한 벤치마크에서 우수한 성능 달성 
- AIME 2024에서 평균 53.3%(15문제 중 8문제) 해결
- 상위 20% 고교생 수학 실력 수준 달성
6. 의의:
- 작은 언어 모델로도 고수준의 수학 추론 가능성 입증
- 자체 진화 방식을 통한 효율적인 성능 향상 방법 제시
- 수학 교육 및 AI 추론 연구에 새로운 방향 제시
이 연구는 작은 규모의 언어 모델도 적절한 방법론을 통해 복잡한 수학적 추론이 가능함을 보여주었으며, 향후 AI의 수학적 추론 능력 향상 연구에 중요한 통찰을 제공했습니다.
https://arxiv.org/pdf/2501.04519

## MS, sLM 'Phi-4' 완전 오픈 소스로 공개. (출처: 뉴스)
-마이크로소프트(MS)가 지난달 출시한 소형언어모델(sLM) '파이-4(Phi-4)'를 완전한 오픈 소스로 공개.
-작지만 기존의 대형언어모델(LLM)과 견줄 만한 성능 자랑.
-허깅페이스에서 가중치가 포함된 모델을 다운로드할 수 있게 됐으며, 상업적 용도로도 자유롭게 활용 가능.
마이크로소프트(MS)가 지난달 출시한 소형언어모델(sLM) '파이-4(Phi-4)'를 완전한 오픈 소스로 공개했다. 이 모델은 수학을 비롯한 추론 능력을 대폭 강화했으며, 기존의 대형언어모델(LLM)과 견줄 만한 성능을 자랑한다.
시타르 샤 MS 선임 연구원은 8일 X(트위터)를 통해 파이-4를 다운로드 가능한 가중치를 포함한 완전한 오픈 소스로 공개한다고 발표했다.
일반적으로 가중치가 공개되지 않으면 모델을 완전한 오픈 소스로 간주하지 않는데, 이는 연구자들이 모델을 수정하거나 자신만의 용도로 최적화하는 데 제약이 있기 때문이다.
파이-4는 지난해 12월 출시됐으며, 그동안은 '애저 AI 파운드리'를 통해 연구 라이선스 계약 하에서만 접근할 수 있었다. 그러나 이제 허깅페이스에서 가중치가 포함된 모델을 다운로드할 수 있게 됐으며, 상업적 용도로도 자유롭게 활용할 수 있다. 사용자는 MS의 허가 없이도 이 모델을 프로젝트에 통합하거나 특정 애플리케이션에 맞게 미세조정할 수 있다.
샤 연구원은 "파이-4가 공개된 이후 많은 관심과 함께 가중치 공개를 요청하는 목소리가 컸다"라며 "일부 사용자는 불법적으로 가중치를 허깅페이스에 업로드하기도 했다"라고 밝혔다.
140억개의 매개변수를 보유한 파이-4는 언어 처리뿐만 아니라 수학 등 복잡한 추론 작업에서도 뛰어난 성능을 발휘하는 것으로 알려져 환영받고 있다. 미국 수학경시대회(AMC) 테스트에서 '라마', '큐원' 등 기존 오픈 소스 모델뿐만 아니라, 'GPT-4o', '클로드 3.5 소네트', '제미나이 1.5' 등 첨단 언어 모델보다 우수한 결과를 기록했다.
## ModernBERT : Smarter, Better, Faster, Longer
- NVIDIA 와 HuggingFace, Answer AI 등 공동 연구진이 최신 모델 최적화를 encoder-only 모델에 적용하여 기존 인코더보다 크게 다중 목표 수행 성능을 향상
- ModernBERT는 2조 개의 토큰과 초기 Bert의 512 토큰 시퀀스에서 크게 개선된 8192 길이의 시퀀스로 학습
- 다양한 분류 작업과 단일 및 다중 벡터 검색을 포함하는 다양한 평가에서, RoBert, MosiacBert 등의 트랜스포머 인코더 기반 모델 중 최고의 성능을 달성
- 특히 문서와 코드 검색에서 뛰어난 결과
- ModernBert는 강력한 다운스트림 성능 외에도 ModernBERT는 가장 빠르고 메모리 효율적인 인코더로, 일반 GPU에서 추론(inference)하도록 설계

[https://huggingface.co/.../tran.../main/model_doc/modernbert](https://huggingface.co/.../tran.../main/model_doc/modernbert)
## 단일 이미지에서 정확하게 방향을 추정하는 Object Anything
홍콩대학교와 Sea AI Lab 등 공동연구진이 단일 이미지 내 객체들의 정확한 방향을 추정하는 모델, Object Anything을 소개하였습니다. 라벨 데이터 부족을 해결하기 위해, 연구자들은 3D 객체의 전면을 주석 처리하고 랜덤 뷰에서 이미지를 렌더링해 200만 개의 주석 이미지 데이터셋을 구축했습니다. Object anything은 3D 방향을 확률 분포로 모델링해 예측하며, 합성 데이터에서 실제 데이터로의 전이를 개선하는 전략을 사용합니다. 이를 통해 렌더링 및 실제 이미지에서 최고 수준의 정확도를 달성하고, 다양한 시나리오에서 뛰어난 제로샷 성능을 보여줍니다. 이 모델은 복잡한 공간 개념 이해, 3D 객체 자세 조정 등 다양한 응용을 강화합니다.
[https://orient-anything.github.io/](https://orient-anything.github.io/)

## AniDoc(HKUST, Ant Group), 참조 캐릭터에 따라 스케치 시퀀스를 컬러 애니메이션으로 자동 변환, 사용자가 캐릭터 이미지와 시작 및 종료 스케치만 제공하면 시간적으로 일관된 애니메이션을 쉽게 만들어 줌. 코드 공개. 
AniDoc: Animation Creation Made Easier (2412, HKUST, Ant Group)

 2D 애니메이션 제작은 캐릭터 디자인, 키프레임 애니메이션, 중간 단계, 채색 등 네 가지 필수 단계를 포함하는 업계 표준 워크플로우를 따릅니다. 우리의 연구는 점점 더 강력해지는 제너레이티브 AI의 잠재력을 활용하여 위 과정의 인건비를 줄이는 데 초점을 맞추고 있습니다. 비디오 확산 모델을 기반으로 하는 AniDoc은 참조 캐릭터 사양에 따라 스케치 시퀀스를 컬러 애니메이션으로 자동 변환하는 비디오 라인 아트 컬러화 툴로 등장했습니다. 이 모델은 대응 매칭을 명시적인 지침으로 활용하여 레퍼런스 캐릭터와 각 라인 아트 프레임 사이의 변형(예: 자세)에 대해 강력한 견고성을 제공합니다. 또한, 이 모델은 중간 과정을 자동화할 수도 있어 사용자가 캐릭터 이미지와 시작 및 종료 스케치만 제공하면 시간적으로 일관된 애니메이션을 쉽게 만들 수 있습니다. 
project : https://yihao-meng.github.io/AniDoc_demo/
paper : https://arxiv.org/abs/2412.14173
code : https://github.com/yihao-meng/AniDoc

## Huggingface, sLM용 추론 기술 ‘테스트-타임 스케일링(test-time scaling)’ 오픈 소스 공개 
sLM이 대형언어모델(LLM)처럼 높은 성능과 메모리가 부족할 때 유용

이를 통해 라마-3.2 1B 모델은 훨씬 큰 8B 모델의 성능을 앞질렀으며, 3B 모델은 심지어 70B 모델보다 더 나은 결과

두개의 모델을 병렬로 실행해야 한다는 조건, 코딩이나 수학처럼 답변을 명확히 평가할 수 있는 문제에서만 성능 발휘

오픈소스 sLM에도 적용할 수 있도록 기술을 개방한 것은 적지 않은 영향, 환각이나 정확도, 비용 문제 등으로 모델 도입을 주저했던 기업에는 도움이 될 수도.

허깅페이스가 오픈 소스 소형언어모델(sLM)의 추론 성능을 향상하는 기술을 공개했다. 오픈AI의 ‘o1’처럼 모델에 추가적인 컴퓨팅 리소스와 시간을 투입해 응답 품질을 높이는 ‘테스트-타임 컴퓨트(Test-Time Compute)’ 방식을 기반

허깅페이스는 최근 sLM이 복잡한 수학과 코딩, 추론 문제를 해결할 수 있도록 추가 컴퓨팅 자원과 시간을 활용하는 ‘테스트-타임 스케일링(test-time scaling)’ 기술을 공개

추론 과정에서 더 많은 자원과 시간을 사용해 어려운 질문에 대한 응답 정확도를 높이는 방식이다. 이를 통해 sLM이 대형언어모델(LLM)처럼 높은 성능을 낼 수 있게 도와주며, LLM을 실행하기에 메모리가 부족할 때 유용하다는 설명

특히 테스트-타임 컴퓨트 기술의 단계별 추론 프로세스를 전부 공개한 것에 주목할 만 하다. 오픈AI는 '생각의 사슬(CoT)' 구조와 같은 내부 작동 방식을 비공개로 유지하고 있지만, 허깅페이스는 지난 8월 발표된 딥마인드의 연구를 기반으로 자체 테스트-타임 컴퓨트 기술을 공개

이 기술은 추론 시 추가적인 컴퓨팅을 사용하는 ‘테스트-타임 스케일링’과 함께 sLM의 응답을 평가하는 보상 모델(reward model), 그리고 답변을 개선하기 위해 경로를 최적화하는 탐색 알고리즘 등으로 구성된다. 

테스트-타임 스케일링의 주요 방식으로는 ▲다수결 투표(majority voting) ▲베스트 오브 N(Best-of-N) ▲가중 베스트 오브 N(Weighted Best-of-N) 등이 있다.
다수결 투표 방식은 동일한 질문을 여러번 보내고 가장 많이 선택된 답을 고르는 방법이다. 간단한 문제에서는 효과적일 수 있지만, 복잡한 문제에서는 한계가 있다.
베스트 오브 N은 여러 답변을 생성하고, 다수결 대신 보상 모델을 사용해 최적의 답을 선택한다. 또 가중 베스트 오브 N은 베스트 오브 N을 발전시킨 형태로, 답변의 일관성을 고려하여 자신감이 높고 자주 나타나는 답을 선택한다.
연구진은 프로세스 보상 모델(PRM)을 사용해 최종 답변뿐만 아니라 답변에 도달하는 과정도 평가했다. 실험 결과, 가중 베스트 오브 N과 PRM을 활용한 '라마-3.2 1B' 는 난이도가 높은 MATH-500 벤치마크에서 '라마-3.2 8B'에 가까운 성능을 보였다.
성능을 더 향상하기 위해 ‘빔 탐색(beam search)’ 알고리즘을 추가했다. 이 방법은 모델이 답변하는 과정을 단계적으로 구분, 각 단계에서 생성된 답변을 탐색 알고리즘이 보상 모델로 평가해 최적의 답을 찾아내는 방식이다.
하지만 빔 탐색은 복잡한 문제에서는 성능을 개선할 수 있지만, 간단한 문제에서는 다른 방식보다 성능이 떨어지는 경향을 보였다. 이를 해결하기 위해 'DVTS(Diverse Verifier Tree Search)'와 '연산 최적화 확장 전략(compute-optimal scaling strategy)'을 추가했다. 
DVTS는 잘못된 추론 경로를 피하고 다양한 응답을 찾아내도록 설계된 빔 탐색의 변형이다. 또 연산 최적화 확장 전략은 문제의 난이도에 따라 최적의 추론 방식을 동적으로 선택한다.
이를 통해 라마-3.2 1B 모델은 훨씬 큰 8B 모델의 성능을 앞질렀으며, 3B 모델은 심지어 70B 모델보다 더 나은 결과를 도출했다.
그러나 허깅페이스는 "테스트-타임 스케일링에는 여전히 한계가 있다"라고 밝혔다.
예를 들어, 실험에서는 PRM으로 훈련된 라마-3.1-8B 모델을 사용했으며, 이 모델은 두개의 모델을 병렬로 실행해야 한다는 조건이 따른다. 또 이 기술은 코딩이나 수학처럼 답변을 명확히 평가할 수 있는 문제에서만 성능을 발휘한다.
하지만 추론의 핵심인 테스트-타임 컴퓨팅을 오픈 소스 sLM에도 적용할 수 있도록 기술을 개방한 것은 적지 않은 영향을 미칠 수 있다는 설명이다.
특히 기업의 오픈 소스 모델 활용이 늘어나는 가운데, 이제까지 환각이나 정확도, 비용 문제 등으로 모델 도입을 주저했던 기업에는 도움이 될 수 있다.
https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute?fbclid=IwY2xjawHYyb9leHRuA2FlbQIxMAABHcHOJZMIH916HhhREOCeBUCY7bRBGVe3FFEKr52px0n8pVMuhfUMElcvvw_aem_cOomg8vK8uUU8_XS1Clu5Q

## Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale (2412, NVIDIA)
(설명, 한글 번역)메시트론은 1024 레벨 좌표 해상도에서 최대 64K 면의 메시를 생성할 수 있는 새로운 자동 회귀 메시 생성 모델로, 현재의 최신 방법보다 훨씬 많은 면 수와 8배 높은 좌표 해상도를 제공합니다. 기존 방식보다 50% 이상 적은 트레이닝 메모리, 2.5배 빠른 처리량, 더 나은 일관성을 제공합니다. 메시트론은 전례 없는 수준의 해상도와 충실도로 정교하고 복잡한 3D 오브젝트의 메시를 생성하여 전문 아티스트가 만든 것과 매우 유사하며 애니메이션, 게임 및 가상 환경을 위한 세부적인 3D 에셋을 더욱 사실적으로 생성할 수 있는 길을 열어줍니다.
site : https://research.nvidia.com/labs/dir/meshtron/
paper : https://arxiv.org/abs/2412.09548
## HunyuanVideo ComfyUI 공식 example & workflow 
현재 오픈소스 AI video에서 큰 관심을 끌고 있는 tencent의 HunyuanVideo 를 ComfyUI 에서 공식 지원하고 그 example 과 workflow 를 공개했네요.
문서 안에 설치법 및 다운로드 받을 파일들 명시되어 있으니 따라서 하면 되고, comfyui 최신으로 업데이트해야 인식됩니다.

많은 메모리가 필요하기 때문에 24GB 인데도 out of memory 나거나 더 작은 메모리 GPU를 사용하고 있다면 resolution(width/height)를 줄여보거나 fp8 또는 city96님이 올려 주신 경량화 모델 HunyuanVideo-gguf 를 고려해 보시기 바랍니다. 

comfyui example/workflow : https://comfyanonymous.github.io/ComfyUI.../hunyuan_video/

HunyuanVideo-gguf : https://huggingface.co/city96/HunyuanVideo-gguf

## NC, 한국어 특화 오픈소스 비전언어모델(VLM) ‘바르코-비전(VARCO-VISION)·벤치마크 5종 공개…“오디오·비디오까지 확대할 것” 
엔씨소프트는 한국어 특화 중소형 오픈 소스 비전언어모델(VLM) 모델 ‘바르코-비전(VARCO-VISION)’과 한국어 멀티모달 벤치마크 5종을 공개했다. 

 VLM은 자연어와 이미지를 활용해 입력값(프롬프트)을 넣을 수 있는 언어모델이다. 현재 오픈 소스로 공개된 대부분 VLM 중 한국어 지원 모델은 소수에 불과하다. 한글과 영어 프롬프트, 이미지 입력 값을 이해할 수 있는 중소형 모델이다.
 대형언어모델(LLM)과 유사한 수준의 언어 능력을 보유하고 있어, LLM과 VLM 두개를 따로 운용하지 않고 단일 모델 만으로 이미지-텍스트 작업과 텍스트 전용 작업을 모두 처리할 수 있다. 특히 한국어 부문에서 동종 크기 모델 중 1위 성능을 보였다고 전했다. ▲OCR ▲그라운딩 ▲레퍼링 등 이미지를 인식 및추론하는 비전 태스크에서도 뛰어난 결과값을 제공한다.이 모델을 활용하면 AI 서비스 개발 기업은 ▲이미지 인식 및 질의응답 ▲이미지 설명 ▲글자인식(OCR) ▲사물 위치 검출(그라운딩) 기능을 활용한 각종 멀티모달 AI 서비스를 개발할 수 있다.

 또 콘텐츠 제작 기업은 이미지 상세 설명을 자동 생성해 콘텐츠 제작 시간을 아끼거나 이미지 내 텍스트 인식을 통해 많은 자료를 빠르게 수집하는 등 기획 업무에 도움을 받을 수 있다고 설명했다. 특히 향후에는 음성과 영상 기능까지 추가, 글로벌 수준의 멀티모달모델(LMM)으로 발전하겠다. 
영미권에서 대표적으로 사용하는 객관식 벤치마크 3종(MM벤치, 시드-벤치, MM스타)과 주관식 벤치마크 1종(라바-인-더-와일드)을 기반으로 새로운 한국어 벤치마크 4종을 구축했다고 전했다. 더불어 한국어 문서, 표, 차트에 대한 이해능력을 검증할 수 있는 ‘K-DTC벤치’ 1종을 새로 추가했다.

## AI Video Model Comparison: Text to Video (source: @HBCoop_)
몇일전 발표된 중국 Tecncent 의  Hunyuan Video가 관심을 받으면서 기존에 Runway Gen-3, Kling AI 1.5, Hailuo MiniMax, Luma Dream Machine 비교가 많았었는데 비교 대상에서 Luma Dream Machine 를 빼고 Hunyuan Video와 비교 하기 시작하네요.  Runway 빼고 모두 중국 서비스.  
- Runway Gen-3
- Kling AI 1.5
- Hailuo MiniMax
- Hunyuan Video (Tencent)
- 
@HBCoop_: Hunyuan Video를 제외한 각 모델에서 동일한 프롬프트를 4번 실행하여 가장 마음에 드는 결과를 선택했습니다. 이 프롬프트는 모션, 말의 팔다리 및 움직임, 물과의 상호 작용으로 인해 어려운 것입니다.

Prompt: Cinematic tracking shot of a white horse galloping through shallow crystal-clear water at sunrise, water splashing in slow motion, majestic mountains in background. Ultra smooth movement, dramatic natural lighting

## Hailuo AI (MiniMax) I2V-01-Live 출시, 2D 일러스트레이션의 애니메이션화 특징. (source: offiicial X @Hailuo_AI)  
Hailuo I2V-01-Live 소개: 정적인 예술을 동적인 걸작으로 바꾸기
라이브는 2D 일러스트레이션에 생명을 불어넣는 방식에 혁신을 가져올 수 있도록 설계된 I2V 제품군의 최신 버전입니다. 향상된 부드러움과 생생한 모션으로 이전과는 전혀 다른 방식으로 캐릭터가 움직이고, 말하고, 빛날 수 있습니다.
안정성과 섬세한 표현에 최적화된 Hailuo I2V-01-Live는 다양한 예술적 스타일을 지원하여 창의적인 표현을 확장하고 비교할 수 없는 유동성과 정교함으로 예술에 생명을 불어넣을 수 있습니다.
@Kabooki_Kai : 2D 애니메이션을 매끄럽게 처리하는 모습이 정말 인상적이었습니다. 이 기술이 다음 단계로 어떻게 발전할지 정말 기대됩니다.
@seiiiiiiiiiiru : Hailuo에서 애니메이션에 특화된 동영상 생성 모델 'I2V-01-Live'가 등장했는데, 일러스트레이션을 전혀 그릴 줄 모르는 아마추어도 몇 시간 만에 상업적으로 활용 가능한 애니메이션 소재를 만들 수 있다.

## PaliGemma 2: 더 강력하고 다재다능해진 오픈 VLM!
PaliGemma 2는 다양한 모델 크기와 입력 해상도를 포괄하는 새로운 오픈 가중치 VLM 제품군으로, 광범위한 캡션, VQA 및 비디오 작업에서 강력한 전송 성능을 제공합니다. 특히 새롭게 추가된 더 큰 변형은 더 큰 컴퓨팅 예산을 가진 사용자에게 PaliGemma에 비해 상당한 개선을 제공합니다.
또한 PaliGemma 2는 음악, 분자 및 의료 영상과 같은 영역을 포함하여 PaliGemma에서 고려된 것 이상의 응용 분야에서 탁월한 성능을 발휘합니다.
https://huggingface.co/papers/2412.03555
SNOOPI: AI 이미지 생성, 안정성과 제어력을 한 번에!
SNOOPI는 1단계 확산 모델에서 안정성과 제어력을 향상시키는 프레임워크입니다. VSD(Variational Score Distillation)를 안정화하기 위해 가이던스 스케일을 동적으로 조정하는 PG-SB(Proper Guidance-SwiftBrush)를 제안합니다.
또한 1단계 확산 모델에 네거티브 프롬프트 안내를 통합하는 최초의 접근 방식인 NASA(Negative-Away Steer Attention)를 도입하여 생성된 이미지에서 원치 않는 특징을 효과적으로 줄입니다. 실험 결과는 SNOOPI가 강력한 기준선을 능가하여 안정성과 효과를 보여줍니다.
https://huggingface.co/papers/2412.02687

## HunyuanVideo(tencent): 중국 텐센트의 오픈소스 비디오 생성 모델. 오픈 소스 모델 중 가장 큰 규모(13b). Runway Gen-3, Luma 1.6, 중국 최고 성능의 동영상 생성 모델들보다 우수하다 주장. GPU RAM은 45G for 544x960, 60GB for 720x1280, Recommended 80GB  
HunyuanVideo: A Systematic Framework For Large Video Generation Model Training (24.12.3 tencent)
 주요 폐쇄형 모델보다 뛰어나지는 않더라도 비슷한 수준의 동영상 생성 성능을 보여주는 새로운 오픈 소스 동영상 기반 모델인 HunyuanVideo를 소개합니다. 훈위안비디오는 데이터 큐레이션, 이미지-비디오 공동 모델 훈련, 대규모 모델 훈련 및 추론을 용이하게 하도록 설계된 효율적인 인프라 등 몇 가지 주요 기능을 통합한 포괄적인 프레임워크를 특징으로 합니다. 또한 모델 아키텍처와 데이터 세트를 확장하는 효과적인 전략을 통해 130억 개 이상의 파라미터로 동영상 생성 모델을 성공적으로 학습시켰으며, 이는 모든 오픈 소스 모델 중 가장 큰 규모입니다.
높은 시각적 품질, 모션 다양성, 텍스트-비디오 정렬 및 생성 안정성을 보장하기 위해 광범위한 실험을 수행하고 일련의 목표 설계를 구현했습니다. 전문가들의 평가 결과에 따르면, HunyuanVideo는 Runway Gen-3, Luma 1.6, 중국 최고 성능의 동영상 생성 모델 3개를 포함한 이전의 최첨단 모델보다 성능이 뛰어납니다. 기초 모델과 그 애플리케이션의 코드와 가중치를 공개함으로써 폐쇄형 소스와 오픈 소스 비디오 기초 모델 간의 격차를 해소하고자 합니다. 이 이니셔티브는 커뮤니티의 모든 사람이 자신의 아이디어를 실험할 수 있도록 지원하여 더욱 역동적이고 활기찬 동영상 제작 생태계를 조성할 것입니다.
We have tested on a single H800/H20 GPU.
Minimum: The minimum GPU memory required is 60GB for 720px1280px129f and 45G for 544px960px129f.
Recommended: We recommend using a GPU with 80GB of memory for better generation quality.
project : https://aivideo.hunyuan.tencent.com/
github : https://github.com/Tencent/HunyuanVideo
huggingface : https://huggingface.co/tencent/HunyuanVideo

## 중국 칭화대, 엣지 디바이스에 최적화된 멀티모달모델 LMM ‘GLM-Edge’ 오픈소스로 공개
GLM-Edge-1.5B 모델은 언어 처리 및 비전 벤치마크에서 훨씬 더 큰 트랜스포머 모델에 필적하는 성과
중국 칭화대 연구진이 언어 처리와 비전 기능을 결합한 경량형 멀티모달모델(LMM)을 공개했다. 클라우드 서버와의 연결 없이 엣지 디바이스에서 실행이 가능하도록 설계, 온디바이스 인공지능(AI) 환경에 최적화된 것이 특징이다.
마크테크포스트는 29일 칭화대 연구진이 에지 디바이스에서 로컬로 실행할 수 있는 온디바이스 AI용 LMM인 ‘GLM-에지(GLM-Edge)’ 시리즈를 오픈 소스로 출시했다고 보도했다.
이 모델군은 15억~50억개의 매개변수를 포함하고 있으며, 언어와 비전 기능을 하나의 모델로 통합해 자원이 제한된 디바이스 환경에서 최적화된 솔루션을 제공한다. 언어 기능은 복잡한 대화를 낮은 지연 시간으로 처리할 수 있고, 비전 기능은 객체 감지와 이미지 캡션 생성 등 다양한 컴퓨터 비전 작업을 실시간으로 지원한다.
GLM-에지 시리즈는 GLM(General Language Model) 아키텍처를 기반으로, 양자화(quantization) 기술과 구조 개선을 통해 에지 환경에서의 활용성을 극대화했다.
또 지식 증류(knowledge distillation)와 프루닝(pruning)을 결합해 학습했다. 이를 통해 모델 크기를 크게 줄이면서도 높은 정확도를 유지했다. 특히 8비트와 4비트 양자화를 적용하여 메모리와 계산 요구량을 줄임으로써, 제한된 자원을 가진 소형 디바이스에서도 실행이 가능하게 설계됐다.
클라우드 서버 없이도 강력한 AI 기능을 발휘할 수 있기 때문에 데이터를 로컬에서 처리해 비용 효율성을 높이는 동시에 개인정보 보호와 보안에도 유리하다. 이는 특히 개인정보 보호, 낮은 지연 시간, 오프라인 작동이 중요한 애플리케이션에서 유용하다.
GLM-Edge-1.5B 모델은 언어 처리 및 비전 벤치마크에서 훨씬 더 큰 트랜스포머 모델에 필적하는 성과를 보였다. 또 키워드 감지나 실시간 비디오 분석 등 에지 환경에서 필요한 작업에서도 우수한 성능을 보여주며, 모델 크기, 지연 시간, 정확도 간의 균형을 성공적으로 달성했다.
현재 GLM-Edge 시리즈는 깃허브와 허깅페이스를 통해 다운로드할 수 있으며, 상업적 용도로도 활용 가능하다.

## 이미지 생성 모델 FLUX를 발표했던 BlackForestLabs 에서 이미지 제어 및 조정할 수 있는 모델 모음인 'FLUX.1 Tools'를 발표하였습니다. 
1.Fill(inpainting/outpainting), 2.Depth(depth map extracting) 3.Canny(canny edges extracting), 4.Redux(image and text mixing regenerating) 

Introducing FLUX.1 Tools
Nov 21, 2024 by BlackForestLabs
오늘, 기본 텍스트-이미지 변환 모델인 FLUX.1에 제어 및 조정 기능을 추가하여 실제 이미지와 생성된 이미지를 수정하고 재창조할 수 있도록 설계된 모델 모음인 FLUX.1 도구를 출시하게 되어 매우 기쁩니다. 출시 시 FLUX.1 도구는 FLUX.1 [dev] 모델 시리즈 내에서 오픈 액세스 모델로 사용할 수 있는 네 가지 기능으로 구성되며, FLUX.1 [pro]를 보완하는 BFL API에서 사용할 수 있습니다
-FLUX.1 Fill: 최첨단 인페인팅 및 아웃페인팅 모델로 텍스트 설명과 바이너리 마스크가 주어지면 실제 이미지와 생성된 이미지를 편집하고 확장 가능. 
-FLUX.1 Depth: 입력 이미지와 텍스트 프롬프트에서 추출한 심도 맵을 기반으로 구조적 안내가 가능하도록 학습된 모델.
-FLUX.1 Canny: 입력 이미지와 텍스트 프롬프트에서 추출한 캐니 에지를 기반으로 구조적 안내가 가능하도록 학습된 모델.
-FLUX.1 Redux: 입력 이미지와 텍스트 프롬프트를 혼합하고 다시 만들 수 있는 어댑터.
이번 릴리스는 다음과 같은 두 가지 약속을 강화합니다: 연구 커뮤니티를 위한 최첨단 개방형 가중치 모델을 제공하는 동시에 API를 통해 동급 최고의 기능을 제공합니다. 저희는 BFL API의 각 도구를 FLUX.1 [pro] 변형으로 출시하며, 추론 코드와 가중치는 지침에 따라 증류된 오픈 액세스 FLUX.1 [dev] 변형으로 사용할 수 있습니다. 또한, 출시되는 모델은 파트너인 fal.ai, Replicate, Together.ai, Freepik 및 krea.ai를 통해 사용할 수 있게 되어 매우 기쁩니다.
다음 섹션에서는 새로운 모델에 대한 자세한 내용과 성능에 대한 분석, 그리고 액세스 방법에 대해 설명합니다. 새로운 도구를 통해 활기찬 Flux 생태계가 어떻게 보완될지 기대가 됩니다.
1. FLUX.1 Fill를 사용한 인페인팅 및 아웃페인팅
FLUX.1 Fill은 Ideogram 2.0과 같은 기존 도구와 AlimamaCreative의 FLUX-Controlnet-Inpainting과 같은 인기 오픈 소스 변형을 능가하는 고급 인페인팅 기능을 도입했습니다. 기존 이미지와 자연스럽게 통합되는 매끄러운 편집이 가능합니다.
또한 FLUX.1 Fill은 아웃페인팅을 지원하여 사용자가 이미지의 원래 테두리를 넘어 이미지를 확장할 수 있습니다.
여기에서 공개적으로 벤치마크를 실시했습니다. 그 결과 Flux.1 Fill [pro]가 다른 모든 경쟁 방법보다 성능이 뛰어나 현재까지 가장 최신의 인페인팅 모델인 것으로 나타났습니다. 두 번째는 Flux.1 Fill [dev]로, 독점 솔루션보다 성능이 뛰어나면서도 추론 효율이 더 높습니다.
Flux.1 Fill [dev]은 Flux 개발자 라이선스에 따라 다음과 같이 사용할 수 있습니다.
-허깅 페이스에서 전체 모델 가중치 사용 가능: [Fill] 
-추론 코드 GitHub에서 사용 가능
Flux.1 Fill [pro]는 [BFL API]에서 사용할 수 있습니다.
2. FLUX.1 Canny/Depth를 사용한 구조적 컨디셔닝
구조 조정은 캐니 에지 또는 깊이 감지 기능을 사용해 이미지 변환 중에 정밀한 제어를 유지합니다. 엣지 또는 뎁스 맵을 통해 원본 이미지의 구조를 보존함으로써 사용자는 핵심 구도를 그대로 유지하면서 텍스트 안내 편집을 할 수 있습니다. 이는 이미지 리텍스처링에 특히 효과적입니다.
여기에서 제공되는 벤치마크 평가에서 FLUX.1 Depth는 Midjourney ReTexture와 같은 독점 모델보다 뛰어난 성능을 발휘합니다. 특히 FLUX.1 Depth [pro]는 더 높은 출력 다양성을 제공하는 반면, FLUX.1 Depth의 dev 버전은 깊이 인식 작업에서 더 일관된 결과를 제공합니다. 캐니 엣지 모델의 경우, 여기 벤치마크에서 FLUX.1 canny [pro]가 동급 최고이며, 그다음으로 FLUX.1 canny [dev]이 그 뒤를 잇습니다.
FLUX.1 Canny/Depth는 최대 성능을 위한 풀 모델과 보다 쉬운 개발을 위한 FLUX.1 [dev] 기반의 LoRA 버전 두 가지로 제공됩니다.
Flux Canny/Depth [dev]은 다음과 같은 플럭스 개발자 라이선스에 따라 사용할 수 있습니다.
-허깅 페이스에서 사용 가능한 전체 모델 가중치: [Depth] [Canny]  
-허깅 페이스에서 사용 가능한 LoRA 가중치: [Depth] [Canny] 
-추론 코드 GitHub에서 사용 가능
Flux.1 Depth/Canny [pro]는 BFL API에서 사용할 수 있습니다.
3. FLUX.1 Redux를 사용한 이미지 변형 및 리스타일링
FLUX.1 Redux는 이미지 변형 생성을 위한 모든 FLUX.1 기본 모델용 어댑터입니다. 입력 이미지가 주어지면 FLUX.1 Redux는 약간의 변형으로 이미지를 재현하여 주어진 이미지를 세분화할 수 있습니다.
프롬프트를 통해 이미지 스타일 변경을 잠금 해제하는 더 복잡한 워크플로에 자연스럽게 통합됩니다. 이미지와 프롬프트를 제공하면 API를 통해 스타일 변경이 가능합니다. 이 기능은 최신 모델인 FLUX1.1 [pro] Ultra에서 지원되며, 입력 이미지와 텍스트 프롬프트를 결합하여 유연한 화면 비율로 고품질의 4메가픽셀 출력을 만들 수 있습니다.
당사의 벤치마크는 FLUX.1 Redux가 이미지 변형에서 최첨단 성능을 달성한다는 것을 보여줍니다.
Flux.1 Redux [dev]은 Flux 개발자 라이선스에 따라 다음과 같이 제공됩니다.
-허깅 페이스에서 사용 가능한 모델 가중치: [Redux] 
-추론 코드 GitHub에서 사용 가능
Flux1.1 [pro] Ultra를 지원하는 Flux1.1 Redux는 BFL API에서 사용할 수 있습니다.
https://blackforestlabs.ai/flux-1-tools/?fbclid=IwY2xjawG4hnRleHRuA2FlbQIxMAABHfAjL7tuVGF54gR9MCSle2AjgAoN_PxjKBqlwal6thziVOkrEyEObYvuHg_aem_b0O4SeTeU4G9A79m2boWXg

##  GPT-4o·클로드 능가하는 코딩 모델 '큐원2.5-코더(Qwen2.5-Coder)' 
알리바바가 새로운 AI 코딩 어시스턴트 '큐원2.5-코더(Qwen2.5-Coder)'를 출시하며 허깅페이스에서 두번째로 인기 있는 모델로 자리 잡았다. 

벤처비트는 12일(현지시간) 알리바바 클라우드가 기존의 '큐원 2.5-코더 1.5B'와 '7B' 버전에 이어 ▲0.5B ▲3B ▲14B ▲32B 매개변수의 4가지 버전을 추가 공개했다고 전했다.
큐원2.5-코더는 12만8000 토큰의 컨텍스트 창과 함께 92개의 프로그래밍 언어를 지원한다. 특히 '큐원2.5-코더-32B-인스트럭트' 모델은 오픈 소스 코딩 모델 중에서 가장 뛰어난 모델로 평가받고 있으며, 'GPT-4o'나 '클로드'를 능가하는 코딩 능력을 갖췄다.

코드 생성 능력을 측정하는 핵심 지표인 휴먼이밸(HumanEval) 벤치마크에서 92.7%, MBPP에서 90.2%를 기록했으며, 실제 코딩 과제를 평가하는 라이브코드벤치(LiveCodeBench)에서는 31.4%의 정확도를 달성했다. 이는 오픈AI와 앤트로픽의 모델을 능가하는 결과다.

또 대부분 AI 코딩 도구가 파이썬이나 자바스크립트와 같은 주요 언어에 집중하는 반면, 큐원2.5-코더는 하스켈(Haskell)과 라켓(Racket)과 같은 특수 언어를 포함해 총 92개 언어에 대해 뛰어난 성능을 발휘했다.

코딩 능력 외에도 일반적인 문제 해결과 수학적 능력에서도 높은 점수를 기록했다.
현재 큐원2.5-코더는 허깅페이스에서 이용 가능하며, 64GB의 맥북 프로 'M2'에서도 실행할 수 있을 만큼 최적화됐다.


## X-Portrait2(bytedance), 얼굴 이미지가 참조 영상을 따라하는 Portrait animation, Lip-sync 모델, Runway Act-one 보다 우수하다 주장.   
X-Portrait 2: Highly Expressive Portrait Animation (2411, ByteDance / Tsinghua University)

이전 작품인 X-Portrait을 기반으로 인물 애니메이션의 표현력을 완전히 새로운 차원으로 끌어올린 X-Portrait 2를 소개합니다. 이를 위해 대규모 데이터 세트에 대한 학습을 통해 입력된 모든 미세한 표현을 암시적으로 인코딩하는 최첨단 표현 인코더 모델을 구축합니다.

그런 다음 이 인코더를 강력한 제너레이티브 확산 모델과 결합하여 유동적이고 표현력이 풍부한 동영상을 생성합니다. 

우리의 X-Portrait 2 모델은 배우의 미묘하고 미세한 표정뿐만 아니라 삐죽거리기, 혀 내밀기, 볼 부풀리기, 찡그리기 등 까다로운 표정까지 전송할 수 있습니다.

또한 생성된 영상에서 높은 충실도의 감정 보존을 달성할 수 있습니다.

site : https://byteaigc.github.io/X-Portrait2/

## ReCapture(Google) 비디오 생성시 카메라 컨트롤이 아닌, 이미 존재하는 비디오에서 새로운 카메라 궤적을 가진 동영상 생성  방법.
ReCapture: Generative Video Camera Controls for User-Provided Videos using Masked Video Fine-Tuning (2411, Google/National University of Singapore)
(설명, 번역글) 최근에는 동영상 모델링의 획기적인 발전으로 생성된 동영상에서 카메라 궤적을 제어할 수 있게 되었습니다. 하지만 이러한 방법은 비디오 모델에 의해 생성되지 않은 사용자 제공 비디오에는 직접 적용할 수 없습니다. 이 백서에서는 사용자가 제공한 단일 동영상에서 새로운 카메라 궤적을 가진 새로운 동영상을 생성하는 방법인 리캡처를 소개합니다. 이 방법을 사용하면 기존의 모든 장면 모션이 포함된 참조 비디오를 매우 다양한 각도에서 시네마틱 카메라 모션으로 재생성할 수 있습니다. 특히, 이 방법을 사용하면 레퍼런스 비디오에서는 관찰할 수 없었던 장면의 일부분을 그럴듯하게 환각화할 수 있습니다. 이 방법은 (1) 멀티뷰 확산 모델 또는 깊이 기반 포인트 클라우드 렌더링을 사용하여 새로운 카메라 궤적으로 노이즈가 있는 앵커 비디오를 생성한 다음 (2) 제안한 마스크 비디오 미세 조정 기술을 사용하여 앵커 비디오를 깨끗하고 시간적으로 일관된 리앵글 비디오로 재생성하는 방식으로 작동합니다.

site : https://generative-video-camera-controls.github.io/
paper : https://arxiv.org/abs/2411.05003

## Hunyuan3D-1.0(Tencent) : text/image to 3D open source model
속도와 품질 간의 인상적인 균형을 달성하여 생성 시간을 크게 단축하는 동시에 제작된 에셋의 품질과 다양성을 유지, 텍스트-이미지 모델, 즉 Hunyuan-DiT가 포함되어 있어 텍스트 및 이미지 조건부 3D 생성을 모두 지원하는 통합 프레임워크. 
Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation 
(설명, 번역) 3D 생성 모델은 아티스트의 워크플로를 크게 개선했지만, 기존의 3D 생성용 확산 모델은 생성 속도가 느리고 일반화가 어렵다는 단점이 있습니다. 이 문제를 해결하기 위해 우리는 라이트 버전과 표준 버전을 포함한 2단계 접근 방식인 Hunyuan3D-1.0을 제안하며, 텍스트 및 이미지 컨디셔닝 생성을 모두 지원합니다. 첫 번째 단계에서는 약 4초 만에 멀티뷰 RGB를 효율적으로 생성하는 멀티뷰 확산 모델을 사용합니다. 이러한 멀티뷰 이미지는 다양한 시점에서 3D 자산의 풍부한 디테일을 캡처하여 단일 뷰에서 멀티뷰 재구성으로 작업을 완화합니다. 두 번째 단계에서는 생성된 멀티뷰 이미지를 바탕으로 3D 자산을 약 7초 만에 빠르고 충실하게 재구성하는 피드 포워드 재구성 모델을 도입합니다. 재구성 네트워크는 멀티뷰 확산으로 인해 발생하는 노이즈와 불일치를 처리하는 방법을 학습하고 조건 이미지에서 사용 가능한 정보를 활용하여 3D 구조를 효율적으로 복구합니다. 우리의 프레임워크에는 텍스트-이미지 모델, 즉 Hunyuan-DiT가 포함되어 있어 텍스트 및 이미지 조건부 3D 생성을 모두 지원하는 통합 프레임워크입니다. 표준 버전에는 라이트 및 기타 기존 모델보다 3배 더 많은 매개변수가 있습니다. 우리의 Hunyuan3D-1.0은 속도와 품질 간의 인상적인 균형을 달성하여 생성 시간을 크게 단축하는 동시에 제작된 에셋의 품질과 다양성을 유지합니다.

site : https://3d.hunyuan.tencent.com/
paper : https://arxiv.org/abs/2411.02293
code : https://github.com/Tencent/Hunyuan3D-1

## Mochi + MochiEdit CoimfyUI (출처 : X @AIWarper)
@AIWarper: "Mochi + MochiEdit 커스텀 노드를 사용하여 4090에서 로컬로 생성. 새로운 비디오 모델 아크가 시작되고 있습니다." 
@AIWarper: "Generated locally on a 4090 with Mochi + MochiEdit custom nodes. New video model arc is beginning and I’m here for it"
- ComfyUI-MochiEdit : https://github.com/logtd/ComfyUI-MochiEdit
   ComfyUI nodes to edit videos using Genmo Mochi
- Mochi, GENMO open source AI Video generation model 설명 : https://www.facebook.com/won.wizard/videos/1750497392155120

## PMRF(Technion–Israel Institute of Technology), 
품질 낮은 얼굴 사진 등을 품질 좋은 사진으로 복원. 이미지 복원 작업에서 이전 방법보다 성능이 우수, 이전의 모델 GFPGAN, CodeFormer 등보다 우수하다 주장. code 등도 공개.
Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration (2410, Technion–Israel Institute of Technology)
(설명, 번역) 사실적인 이미지 복원 알고리즘은 일반적으로 왜곡 측정(예: PSNR, SSIM)과 지각 품질 측정(예: FID, NIQE)에 의해 평가되며, 지각 품질을 손상시키지 않으면서 가능한 가장 낮은 왜곡을 달성하는 것이 목표입니다. 이러한 목표를 달성하기 위해 현재의 방법은 일반적으로 사후 분포에서 샘플링하거나 왜곡 손실(예: MSE)과 지각 품질 손실(예: GAN)의 가중치 합을 최적화하려고 시도합니다. 이전 연구와 달리 이 논문에서는 특히 완전 지각 지수라는 제약 조건, 즉 재구성된 이미지의 분포가 기준 진실 이미지의 분포와 동일한 조건에서 MSE를 최소화하는 최적의 추정자에 대해 다룹니다. 최근의 이론적 결과에 따르면 이러한 추정기는 사후 평균 예측(MMSE 추정치)을 실측 이미지의 분포로 최적으로 변환하여 구축할 수 있습니다. 이 결과에서 영감을 얻어 이 최적 추정치에 근사치를 구하는 간단하면서도 매우 효과적인 알고리즘인 후방 평균 정류 흐름(PMRF)을 소개합니다. 특히 PMRF는 먼저 후방 평균을 예측한 다음, 원하는 최적 전송 맵에 근사한 정류된 흐름 모델을 사용하여 결과를 고품질 이미지로 변환합니다. 우리는 PMRF의 이론적 유용성을 조사하고 다양한 이미지 복원 작업에서 이전 방법보다 일관되게 성능이 우수하다는 것을 입증합니다.
site : https://pmrf-ml.github.io/
paper : https://arxiv.org/abs/2410.00418
code : https://github.com/ohayonguy/PMRF
demo : https://huggingface.co/spaces/ohayonguy/PMRF

### PMRF, Image restoration test
품질 낮은 얼굴 사진 등을 품질 좋은 사진으로 복원해주며, 이전의 모델 GFPGAN, CodeFormer 등보다 우수하다 주장하는 PMRF 테스트해 보았습니다.
test site : https://huggingface.co/spaces/ohayonguy/PMRF
PMRF 설명 : https://www.facebook.com/won.wizard/posts/pfbid033f6g8MyUH9gSC22pcjvUvooRmQ2W94nfnc4ZfMFW8secM3GwYwM5goQbKnPHsZCel

## Flux Controlnet Upscaler(Jasperai) test (include comfyui workflow)
Flux Controlnet Upscaler : This is Flux.1-dev ControlNet for low resolution images developed by Jasper research team.
flux comfyui workflow : https://github.com/.../blob/main/flux.1-dev-upscaler.json
controlnet upscaler model download : https://huggingface.co/jas.../Flux.1-dev-Controlnet-Upscaler

## 오픈AI, 이미지 생성 속도 50배 높이는 모델 개발..."실시간 생성 AI 콘텐츠 개념 가까워" 
단 2단계만으로 생성 가능 연속 시간 일관성 모델(Continuous-Time Consistency Models) ‘sCM’, 기존 확산 모델(Diffusion Model) 보다 50배 빠르게 이미지 생성, Dalle-4 기반 모델 가능성도.
오픈AI가 기존 확산 모델(Diffusion Model) 보다 50배 빠르게 이미지를 생성하는 새로운 모델을 내놓았다. 이 정도 속도는 '실시간 생성'에 가까운 것으로, 향후 미디어와 콘텐츠 산업에도 영향을 미칠 수 있다는 반응이다.
벤처비트는 23일 오픈AI의 연구원 챙 루와 양 송이 단 2단계만으로 고품질 샘플을 생성할 수 있는 새로운 유형의 연속 시간 일관성 모델 ‘sCM’에 관한 논문을 아카이브에 게재했다고 보도했다.
샘플을 생성하기 위해 수백단계의 노이즈 제거가 필요한 기존 확산 모델과 달리, sCM은 1~2단계 만에 노이즈를 고품질 샘플로 직접 변환 계산 비용과 시간을 줄여준다.
이 모델은 인공지능(AI)이 이미지, 비디오, 오디오를 포함한 멀티미디어를 생성하는 속도를 기존 확산 모델보다 50배나 높여서 일반 확산의 경우 5초가 넘는 데 비해 거의 10분의 1초 만에 이미지를 생성한다. 실제로 15억 매개변수의 가장 큰 sCM 모델의 경우 단일 A100 GPU에서 단 0.11초 만에 샘플을 생성할 수 있다. 이를 통해 실시간 생성 AI 애플리케이션이 훨씬 더 실행 가능해졌다.
sCM은 불과 15억 매개변수의 규모로 최고의 확산 모델에 필적하는 샘플 품질을 유지하며, 이미지넷 512×512에서 1.88의 FID 점수를 달성했다. 이는 훨씬 더 많은 계산 자원을 필요로 하는 확산 모델과 비교해 유사한 결과를 얻는 데 10% 이내의 계산 자원만을 사용한다.
확산 모델은 사실적인 이미지, 3D 모델, 오디오 및 비디오를 제작하는 데 뛰어난 결과를 제공했지만, 종종 수십개에서 수백개의 순차적 단계가 필요한 샘플링의 비효율성으로 인해 실시간 애플리케이션에는 적합하지 않았다.
이 기술은 향후 오픈AI의 실시간 AI 이미지 생성 모델의 기반이 될 것이라는 전망이 나온다. 또 sCM이 '달리-4'의 기반 모델이 될 것이라는 설명이다.
이 모델은 X(트위터)에서 일부 연구자의 호응을 끌어내고 있다. 특히 0.11초라는 생성 시간은 '실시간 생성'이라는 개념을 만들 수 있다는 반응이다.
마티유 로이라는 연구자는 "이는 생성 AI 콘텐츠의 미래를 혁신할 수 있다"라며 "더 빠르고 효율적인 알고리즘은 헬스케어부터 엔터테인먼트까지 다양한 산업에서 새로운 가능성을 열어줄 것"이라고 반응했다.
한편 논문의 저자인 양 송은 앞서 2023년에 오픈AI의 전 수석 과학자인 일리야 수츠케버와 함께 발표한 논문에서 ‘일관성 모델’이라는 개념을 처음 제안했다. 이 모델은 ‘같은 궤적 상의 점들이 동일한 초기 지점에 매핑된다’는 아이디어를 담고 있다.
https://arxiv.org/pdf/2410.11081

## MS, 트랜스포머 성능 개선하는 새로운 LLM 아키텍처, Diff Transformer 공개
'트랜스포머' 기반 대형언어모델(LLM)의 긴 컨텍스트 정보 검색 기능을 개선하는 새로운 아키텍처가 나왔다. 
벤처비트는 16일(현지시간) 마이크로소프트(MS)와 칭화대학교 연구진이 관련 컨텍스트에 대한 어텐션(attention)을 증폭하고 노이즈를 걸러내 성능을 개선하는 새로운 아키텍처 ‘차등 트랜스포머(Diff Transformer)’에 관한 논문을 아카이브에 게재했다고 보도했다.
트랜스포머 아키텍처는 대부분 LLM의 기반이다. 어텐션 메커니즘을 사용해 입력 텍스트 내 토큰이 출력 생성에 미치는 중요도를 평가하는 방식이다. 어텐션 메커니즘은 벡터 값을 확률 분포로 정규화하는 '소프트맥스(softmax)' 함수를 사용해 입력 시퀀스의 토큰에 어텐션 점수를 할당한다.
연구자들은 차등 트랜스포머를 다양한 언어 모델링 작업에서 평가했다.

그 결과, 기존 트랜스포머 아키텍처를 능가하는 것으로 나타났다. 1조개의 토큰으로 훈련한 30억 매개변수의 차등 트랜스포머는 비슷한 크기의 트랜스포머 모델에 비해 몇 % 개선을 보였다.
또 다양한 모델 크기와 훈련 데이터셋 크기를 활용한 추가 실험에서, 기존 트랜스포머와 유사한 성능을 달성하기 위해 65%의 매개변수나 훈련 토큰만 필요한 것으로 나타났다.
연구진은 "증가하는 컨텍스트 길이를 사용하는 데 특히 효과적"이라고 밝혔다. 차등 트랜스포머 기반 모델은 중요한 정보 검색이나 환각 완화, 컨텍스트 내 학습에서 상당한 개선을 보였다고 강조했다.
현재 차등 트랜스포머의 코드는 깃허브에서 사용할 수 있다.

## 엔비디아, 오픈AI·앤트로픽 능가하는 LLM 공개, Llama-3.1-Nemotron-70B-Instruct, 모델 중심 생태계 구축하나
이달 초 대형멀티모달모델(LMM)을 공개하며 오픈AI 등과 모델 경쟁을 선언한 엔비디아가 이번에는 대형언어모델(LLM)을 내놓았다. 이번에는 벤치마크에서 오픈AI의 'GPT-4o'와 앤트로픽의 '클로드 3.5 소네트'를 제치고 최고 점수를 기록했다고 밝혔다.
벤처비트는 16일 엔비디아가 별 홍보 없이 허깅페이스를 통해 'Llama-3.1-Nemotron-70B-Instruct'를 출시했다고 보도했다. 이 모델은 엔비디아 전용 플랫폼에서 무료로 사용해 볼 수 있다.
엔비디아는 모델 개발에 인간 피드백을 통한 강화 학습(RLHF)과 고품질 데이터셋을 사용, 라마 3.1을 미세조정했다고 밝혔다.
추가 프롬프트나 특수 토큰 없이 복잡한 쿼리를 처리할 수 있는 능력도 강조했다. 선보인 데모에서는 'strawberry에는 r이 몇개 있나요'라는 질문에 정확하게 답했다.
특히 인간 선호도 평가인 '아레나 하드벤치'에서 85.0을 기록했으며, '알파카이벨 2 LC(AlpacaEval 2 LC)'에서 57.6, 'GPT-4- 터보 MT-벤치'에서 8.98 등 주요 평가에서 오픈AI의 GPT-4o와 앤트로픽의 클로드 3.5 소네트를 제치고 최고 점수를 기록했다.
이대로라면 현존 최강의 성능을 갖춘 모델이라는 말이다.
이에 앞서 지난 1일에는 'NVLM-D-72B'이라는 엔비디아의 오픈 소스 LMM이 화제가 됐다. 이 모델 역시 대부분 벤치마크에서 GPT-4o나 클로드 3.5 소네트, '제미나이 1.5 프로', '라마 3-V 405B' 등과 대등하거나 높은 성능을 보인 바 있다.
이 때문에 엔비디아가 본격적으로 프론티어 모델 경쟁에 뛰어드는 것이 아니냐는 추측도 나왔다. 물론 이번에 출시한 모델은 자체 개발이 아니라, 라마 3.1을 베이스로 성능을 개선한 모델이다.
하지만 엔비디아의 인프라에 최적화된 모델을 무료로 제공하면, 이는 CUDA를 통해 생태계를 구축한 것과 비슷한 효과를 낼 수 있다는 분석이다. 특히, 엔비디아는 성능과 함께 비용 효율적인 면도 강조하고 있다.
https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF


## CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos
https://go.fb.me/xiyc63
Demo on Hugging Face ➡️ https://go.fb.me/yzuqd0

Building on the popular release of CoTracker, we're introducing CoTracker3, which includes a new tracking model and a new semi-supervised training recipe. Available in online and offline variants, the new model demonstrates impressive tracking results where points can be tracked for extended durations even when they are occluded or leave the field of view. CoTracker3 achieves state-of-the-art and outperforms all recent point tracking approaches on standard benchmarks — often by a substantial margin.
We've released the research paper, code and a demo on Hugging Face — along with models available under an A-NC license to support further research in this space.

## EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation

초록
현재의 자동 회귀 메시 생성 방법은 불완전성, 불충분한 디테일, 낮은 일반화율 등의 문제를 안고 있습니다. 이 논문에서는 512의 공간 해상도에서 최대 4,000개의 면을 가진 고품질 3D 메시를 생성할 수 있는 자동 회귀 자동 인코더(ArAE) 모델을 제안합니다. 

새로운 메시 토큰화 알고리즘을 도입하여 삼각형 메시를 1D 토큰 시퀀스로 효율적으로 압축하여 훈련 효율성을 크게 향상시킵니다. 또한, 가변 길이의 삼각형 메시를 고정 길이의 잠재 공간으로 압축하여 잠재 확산 모델을 훈련함으로써 일반화를 개선할 수 있습니다. 

광범위한 실험을 통해 포인트 클라우드 및 이미지 조건부 메시 생성 작업 모두에서 모델의 우수한 품질, 다양성 및 일반화 기능을 입증했습니다.

프로젝트 https://research.nvidia.com/labs/dir/edgerunner/
논문 https://arxiv.org/pdf/2409.18114

## OpenAI 음성인식 오프소스 Whisper V3 Turbo 공개.
Whisper Large보다 8배, Whisper Medium보다 4배, Whisper Small 모델보다 2배 빠른 속도. 속도가 빠르면서도 성능이 크게 저하되지 않음.
Whisper V3 Turbo는 809M의 파라미터를 갖추고 있으며, 다국어 지원(한국어 포함 99개 언어).
https://huggingface.co/openai/whisper-large-v3-turbo
Whisper는 자동 음성 인식(ASR) 및 음성 번역을 위한 최첨단 모델로, OpenAI의 Alec Radford 등이 작성한 논문 '대규모 약한 감독을 통한 강력한 음성 인식'에서 제안되었습니다. 5백만 시간 이상의 레이블이 지정된 데이터로 학습된 Whisper는 제로 샷 환경에서 많은 데이터 세트와 도메인에 일반화할 수 있는 강력한 능력을 보여줍니다.
Whisper large-v3-turbo는 기존 Whisper large-v3의 미세 조정 버전입니다. 즉, 디코딩 레이어 수가 32개에서 4개로 줄어든 것을 제외하면 완전히 동일한 모델입니다. 결과적으로 이 모델은 약간의 품질 저하를 감수하고도 훨씬 더 빨라졌습니다.

## "Training Language Models to Self-Correct via Reinforcement Learning"
이 논문은 LLM이 강화 학습을 기반으로 외부 피드백없이 자체적으로 오류를 인식하고 수정하며 학습하는 방식입니다.

🔧 사용 모델: google/gemma-2-2B-it

📊 데이터셋: 연구 목적에 맞춰 직접 설계 및 제작

이 구현을 통해 강화학습을 활용한 언어 모델의 자기 교정 능력 향상 가능성을 느껴볼 수 있었습니다. runpod에서 간단하게 학습해보실 수 있도록 코드랑 데이터 모두 공개해 놓았습니다.
깃허브 링크 : [https://github.com/daje0601/Google_SCoRe](https://github.com/daje0601/Google_SCoRe)

# OpenAI Strawberry(o1) 

OpenAI Docs
- [https://platform.openai.com/docs/guides/reasoning](https://platform.openai.com/docs/guides/reasoning)
- <img src="https://github.com/user-attachments/assets/b165cb20-9202-4951-8783-6b2f7e0d6071" width="600px"> 

## Blogs
- [OpenAI] [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)
- [OpenAI] [OpenAI o1-mini Advancing cost-efficient reasoning](https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning)
- [OpenAI] [Finding GPT-4’s mistakes with GPT-4](https://openai.com/index/finding-gpt4s-mistakes-with-gpt-4/)
- [Tibor Blaho] [Summary of what we have learned during AMA hour with the OpenAI o1 team](https://twitter-thread.com/t/1834686946846597281)
- [Nathan Lambert] [OpenAI’s Strawberry, LM self-talk, inference scaling laws, and spending more on inference](https://www.interconnects.ai/p/openai-strawberry-and-inference-scaling-laws)
- [Nathan Lambert] [Reverse engineering OpenAI’s o1](https://www.interconnects.ai/p/reverse-engineering-openai-o1)

## Twitter
- [OpenAI Developers](https://x.com/OpenAIDevs/status/1834608585151594537)
- <img src="https://github.com/user-attachments/assets/4670514c-e6fa-474f-abea-c3f6ad01e41a" width="300px">
- <img src="https://github.com/user-attachments/assets/b390ccea-9773-4a96-ba02-40d917473402" width="300px">
- <img src="https://github.com/user-attachments/assets/88896f70-017d-4520-ac56-370a023cfe45" width="300px">
- <img src="https://github.com/user-attachments/assets/fbbf78e4-d34c-4b7b-8163-f8c7288f56a6" width="300px">
- <img src="https://github.com/user-attachments/assets/cb1cc1e6-35d4-4567-891a-4e5aca8fa175" width="300px">
- <img src="https://github.com/user-attachments/assets/d3fd109b-0c97-4a94-931e-919b3b2f75f4" width="300px">

### Relevant Paper from OpenAI o1 [contributors](https://openai.com/openai-o1-contributions/)
```
format:
- [title](paper link) [links]
  - author1, author2, and author3...
  - publisher
  - code
  - experimental environments and datasets
```

- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
  - Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
- [Generative Language Modeling for Automated Theorem Proving](https://arxiv.org/abs/2009.03393)
  - Stanislas Polu, Ilya Sutskever
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
  - Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
  - Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe
- [LLM Critics Help Catch LLM Bugs](https://arxiv.org/abs/2407.00215)
  - Nat McAleese, Rai Michael Pokorny, Juan Felipe Ceron Uribe, Evgenia Nitishinskaya, Maja Trebacz, Jan Leike
- [Self-critiquing models for assisting human evaluators](https://arxiv.org/pdf/2206.05802) 
  - William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, Jan Leike
- [Scalable Online Planning via Reinforcement Learning Fine-Tuning](https://arxiv.org/abs/2109.15316)
  - Arnaud Fickinger, Hengyuan Hu, Brandon Amos, Stuart Russell, Noam Brown.

### 2024 : Relevant Paper from OpenAI o1
- [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733)
  - Evan Wang, Federico Cassano, Catherine Wu, Yunfeng Bai, Will Song, Vaskar Nath, Ziwen Han, Sean Hendryx, Summer Yue, Hugh Zhang
- [An Empirical Analysis of Compute-OptimaInference for Problem-Solving with LanguageModels](https://arxiv.org/abs/2408.00724)
  - Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, Yiming Yang
- [Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling](https://www.arxiv.org/abs/2408.16737)
  - Hritik Bansal, Arian Hosseini, Rishabh Agarwal, Vinh Q. Tran, Mehran Kazemi
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
  - Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240)
  - Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, Rishabh Agarwal
- [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)
  - Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)
  - Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini
- [Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning](https://arxiv.org/abs/2406.14283)
  - Chaojie Wang, Yanchen Deng, Zhiyi Lyu, Liang Zeng, Jujie He, Shuicheng Yan, Bo An
- [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394)
  - Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, Wanli Ouyang
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
  - Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston
- [Uncertainty of Thoughts: Uncertainty-Aware Planning Enhances Information Seeking in Large Language Models](https://arxiv.org/abs/2402.03271)
  - Zhiyuan Hu, Chumin Liu, Xidong Feng, Yilun Zhao, See-Kiong Ng, Anh Tuan Luu, Junxian He, Pang Wei Koh, Bryan Hooi
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
  - Eric Zelikman, Georges Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman
  - https://github.com/ezelikman/quiet-star
- [Advancing LLM Reasoning Generalists with Preference Trees](https://arxiv.org/abs/2404.02078)
  - Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan et al.
- [Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing](https://arxiv.org/abs/2404.12253)
  - Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, and Dong Yu.
- [AlphaMath Almost Zero: Process Supervision Without Process](https://arxiv.org/abs/2405.03553)
  - Guoxin Chen, Minpeng Liao, Chengxi Li, Kai Fan.
- [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search](https://arxiv.org/abs/2406.03816)
  - Dan Zhang, Sining Zhoubian, Yisong Yue, Yuxiao Dong, and Jie Tang.
- [Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning](https://arxiv.org/abs/2405.00451)
  - Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P. Lillicrap, Kenji Kawaguchi, Michael Shieh.
- [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875)
  - Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma.
- [ReFT: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967)
  - Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, Hang Li
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/pdf/2402.10200)
  - Xuezhi Wang, Denny Zhou
 
### 2023 : Relevant Paper from OpenAI o1
- [Training Chain-of-Thought via Latent-Variable Inference](https://arxiv.org/pdf/2312.02179)
  - Du Phan, Matthew D. Hoffman, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, Rif A. Saurous
- [Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training](https://arxiv.org/abs/2309.17179)
  - Xidong Feng, Ziyu Wan, Muning Wen, Stephen Marcus McAleer, Ying Wen, Weinan Zhang, Jun Wang
- [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.14992)
  - Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, Zhiting Hu
- [Don’t throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding](https://arxiv.org/abs/2309.15028)
  - Liu, Jiacheng, Andrew Cohen, Ramakanth Pasunuru, Yejin Choi, Hannaneh Hajishirzi, and Asli Celikyilmaz.
- [Certified reasoning with language models](https://arxiv.org/pdf/2306.04031)
  - Gabriel Poesia, Kanishk Gandhi, Eric Zelikman, Noah D. Goodman     

### 2022 : Relevant Paper from OpenAI o1 
- [Chain of Thought Imitation with Procedure Cloning](https://arxiv.org/abs/2205.10816)
  - Mengjiao Yang, Dale Schuurmans, Pieter Abbeel, Ofir Nachum.
- [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
  - Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman
    
### 2021 : Relevant Paper from OpenAI o1
- [Scaling Scaling Laws with Board Games](http://arxiv.org/abs/2104.03113)
  - Andy L. Jones.
- [Show Your Work: Scratchpads for Intermediate Computation with Language Models](https://arxiv.org/pdf/2112.00114)
  - Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, Augustus Odena
 
### 2017 : Relevant Paper from OpenAI o1
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815v1)
  - David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, Demis Hassabis. 

### Evaluation of OpenAI o1
- [AryanDLuffy] [codeforces](https://codeforces.com/blog/entry/133962)

## LLM Evaluation / Benchmark
- [evals](https://github.com/openai/evals) (`OpenAI`) ![](https://img.shields.io/github/stars/openai/evals.svg?style=social) Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.

## LLM Training / Finetuning
- [xtuner](https://github.com/InternLM/xtuner) (`InternLM`) ![](https://img.shields.io/github/stars/InternLM/xtuner.svg?style=social) An efficient, flexible and full-featured toolkit for fine-tuning LLM (InternLM2, Llama3, Phi3, Qwen, Mistral, ...)

- [litGPT](https://github.com/Lightning-AI/litgpt) (`LightningAI`) ![](https://img.shields.io/github/stars/Lightning-AI/litgpt.svg?style=social) 20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale.

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (`NVIDIA`) ![](https://img.shields.io/github/stars/NVIDIA/Megatron-LM.svg?style=social) Ongoing research training transformer models at scale

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) ![](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory.svg?style=social) A WebUI for Efficient Fine-Tuning of 100+ LLMs (ACL 2024)

- [nanoGPT](https://github.com/karpathy/nanoGPT) (`karpathy`) ![](https://img.shields.io/github/stars/karpathy/nanoGPT.svg?style=social) The simplest, fastest repository for training/finetuning medium-sized GPTs.
 
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) ![](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory.svg?style=social) A WebUI for Efficient Fine-Tuning of 100+ LLMs (ACL 2024)

- [evals](https://github.com/openai/evals) (`OpenAI`) ![](https://img.shields.io/github/stars/openai/evals.svg?style=social) Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.
  
- [nanotron](https://github.com/huggingface/nanotron) (`HuggingFace`) ![](https://img.shields.io/github/stars/huggingface/nanotron.svg?style=social) Minimalistic large language model 3D-parallelism training

## LLM Data Preprocessing
- [NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) (`NVIDIA`) ![](https://img.shields.io/github/stars/NVIDIA/NeMo-Curator.svg?style=social) Scalable toolkit for data curation

- [data-juicer](https://github.com/modelscope/data-juicer) (`ModelScope`) ![](https://img.shields.io/github/stars/modelscope/data-juicer.svg?style=social) A one-stop data processing system to make data higher-quality, juicier, and more digestible for (multimodal) LLMs!

- [datatrove](https://github.com/huggingface/datatrove) (`HuggingFace`) ![](https://img.shields.io/github/stars/huggingface/datatrove.svg?style=social) Freeing data processing from scripting madness by providing a set of platform-agnostic customizable pipeline processing blocks.

- [dataverse](https://github.com/UpstageAI/dataverse) (`Upstage`) ![](https://img.shields.io/github/stars/UpstageAI/dataverse.svg?style=social) The Universe of Data. All about data, data science, and data engineering

- [NeMo-Curator](https://github.com/NVIDIA/NeMo-Curator) (`NVIDIA`) ![](https://img.shields.io/github/stars/NVIDIA/NeMo-Curator.svg?style=social) Scalable toolkit for data curation

- [dps](https://github.com/EleutherAI/dps) (`EleutherAI`)![](https://img.shields.io/github/stars/EleutherAI/dps.svg?style=social) Data processing system for polyglot

## 서울과기대 MLP 연구실에서 Bllossom-405B preview를 공개
1. 사전학습에 대한 영향이 미미함: 워낙 큰 모델이라 약간의 추가 사전학습을 진행하면 오히려 성능이 하락합니다. 학습량을 늘리면 저희처럼 돈을 태워야하는데 성능향상이 아주 작습니다.
2. 405B의 실제 성능 (벤치마크 말고)이 정말 GPT4에 범접하는가? 네 실제 사용해보면 초기 GPT4 1월 버전과 거의 흡사합니다. 요즘 좋은 소형모델의 점수가 GPT에 근접하는데, 실제 사용해보면 실망스러울겁니다. 이건 전혀 그렇지 않습니다.
3. Bllossom 405B의 한국어 벤치성능은? LogicKor 9점대, 한국어 MT-Bench SOTA 등을 보이고 있습니다.

- Llama3.1-405B-Inst 대비 5~10% 한국어 성능이 향상 되었습니다 (single turn 기준).
- Llama3.1의 영어 성능을 전혀 손상시키지 않은 완전한 Bilingual 모델입니다.
- 기존 모델 대비 자연스럽고 친절한 한국어 문장을 생성합니다.
- 인간평가, GPT평가(MT-Bench, LogicKor 9점 등) 결과 GPT4와 유사하거나 약간 낮은 성능을 보여줍니다.

https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B
https://huggingface.co/Bllossom/llama-3.1-Korean-Bllossom-405B-GGUF
https://huggingface.co/Bllossom/llama-3.1-Korean-Bllossom-405B

## 일본어 LLM 22B
CyberAgent에서 일본어 LLM 22B 을 공개했습니다. 

기존 모델을 베이스로 사용하지 않고 개발한 225억 파라미터의 CyberAgentLM3-22B-Chat입니다.

LLM의 일본어 능력을 평가하는 일본어LLM 리더보드에서 70B 파라미터의 Meta-Llama-3-70B-Instruct와 동등한 성능을 보였고, 그래서 오픈 일본어 LLM으로는 톱클래스의 성능입니다

모델은 상용 이용 가능한 Apache License 2.0입니다. 

링크 : https://huggingface.co/cyberagent/calm3-22b-chat

## 한국어 형태소 분석기 Kiwi
Kiwi가 0.18.0로 업데이트되었습니다. 
이번 업데이트에서는 외국어 문자와 이모지 지원 등 비 한국어 텍스트에 대한 편의성 기능이 주로 강화되었습니다.
https://github.com/bab2min/Kiwi/

# NYPL

## CM3leon 모델의 weight를 공개
https://ai.meta.com/blog/meta-fair-research-new-releases/

### CM3leon 모델 소개.
https://ai.meta.com/blog/generative-ai-text-images-cm3leon/
GPT-4o 같은 텍스트-이미지에 걸친 멀티모달 투 멀티모달 모델이다. 
텍스트-이미지 상호 생성 부분을 구현하기 위해 멀티모달 입력을 토크나이저 레벨에서 통합해서, 
멀티모달로 넣고 모델에서 한 번에 멀티모달로 빼는 식으로 다양한 종류의 멀티모달 태스크를 가능하게 한다. 

## Toy Models of Superposition
https://transformer-circuits.pub/2022/toy_model/index.html

## Open Synthetic Data Generation Pipeline for Training Large Language Models
The Nemotron-4 340B instruct model lets you generate high-quality data and then the reward model (also released) can filter out data on several attributes.

https://blogs.nvidia.com/.../nemotron-4-synthetic-data.../

Research paper: https://research.nvidia.com/publi.../2024-06_nemotron-4-340b
arXiv : https://arxiv.org/abs/2406.08673

## Nemotron4_340B
2월에 공개되었던 NVIDIA Nemotron-4의 340B 버전이 Base 모델, Instruct 모델 그리고 Reward 모델이 공개되었습니다. 오픈소스 규약 관점에서 모델 수정, 배포, 결과물 활용까지 폭넓게 활용 가능한 형태입니다. 
8조개 토큰에 pretraining 후 1조개를 continued training 해서 총 9조개 토큰에 학습 했네요. Alignment 를 위한 데이터는 대부분 (98% 넘게) 합성을 통해서 만들어 냈다고 합니다.
FP8로 인퍼런스 할때는 8xH100 DGX 1 노드로 돌아가게 만들었다고 하네요. BF16으로 한다면 H200 1노드, H100 이랑 A100은 2노드라고 합니다. 
예시로 데이터 증강용으로 많이 활용하라고 하는데 최대 토큰길이도 4K 밖에 되지 않아 뭔가 서비스 어플리케이션 향으로 쓰기엔 부담스럽긴 하네요. 340B를 리얼타임에 쓰기도 어렵겠구요.
그래도 오픈소스 모델의 다양성이 늘어나는 부분은 의미가 있겠습니다. 
프로젝트 페이지: https://research.nvidia.com/publi.../2024-06_nemotron-4-340b
허깅페이스: https://huggingface.co/nvidia/Nemotron-4-340B-Base

## Teaching LLMs to Express Confidence
https://x.com/omarsar0/status/1797682549608833477
https://arxiv.org/abs/2405.20974

##  Advancing Multimodal Medical Capabilities of Gemini

### 요약: 
많은 임상 작업에는 일반적으로 범용 대형 다중 모드 모델에서는 볼 수 없는 의료 이미지, 유전체학과 같은 특수 데이터에 대한 이해가 필요합니다.

Gemini의 다중 모드 모델을 기반으로 Gemini의 핵심 기능을 계승하고 2D 및 3D 방사선학, 조직병리학, 안과학, 피부과 및 게놈 데이터를 통한 미세 조정을 통해 의료용으로 최적화된 새로운 Med-Gemini 제품군 내의 여러 모델을 개발합니다. 

Med-Gemini-2D는 전문가 평가를 기반으로 AI 기반 흉부 엑스레이(CXR) 보고서 생성을 위한 새로운 표준을 설정합니다.

이는 두 개의 개별 데이터 세트에서 이전 최고 결과를 1%와 12%의 절대 마진으로 초과합니다. 

여기서 57%와 12% 정상 사례에 대한 AI 보고서의 96%, 비정상 사례에 대한 43%, 65%가 원래 방사선 전문의의 보고서와 "동등하거나 더 나은" 것으로 평가됩니다.

우리는 Med-Gemini-3D를 사용하여 3D 컴퓨터 단층촬영(CT) 볼륨에 대한 최초의 대규모 다중 모드 모델 기반 보고서 생성을 시연합니다.

AI 보고서의 53%는 임상적으로 허용 가능한 것으로 간주되지만 전문 방사선 전문의 보고 품질을 충족하려면 추가 연구가 필요합니다.

보고서 생성 외에도 Med-Gemini-2D는 CXR 시각적 질문 답변(VQA)에서 이전 최고 성능을 능가하고 CXR 분류 및 방사선학 VQA에서 우수한 성능을 발휘하여 20개 작업 중 17개 작업에서 SoTA 또는 기준선을 초과합니다. 

조직병리학, 안과학, 피부과 이미지 분류에서 Med-Gemini-2D는 20개 작업 중 18개 작업에서 기준선을 능가하고 작업별 모델 성능에 접근합니다.

영상 촬영 외에도 Med-Gemini-Polygenic은 질병 위험 예측을 위한 표준 선형 다유전성 위험 점수 기반 접근 방식을 능가하며, 훈련된 적이 없는 유전적으로 연관된 질병을 일반화합니다. 

안전이 중요한 의료 영역에서는 추가 개발과 평가가 필요하지만, 우리의 결과는 광범위한 의료 작업에 걸쳐 Med-Gemini의 잠재력을 강조합니다.

### link
arXiv: https://arxiv.org/abs/2405.03162
Browse: https://browse.arxiv.org/pdf/2405.03162.pdf

PDF: https://arxiv.org/pdf/2405.03162.pdf  

arXiv-vanity: https://www.arxiv-vanity.com/papers/2405.03162 
Paper page: https://huggingface.co/papers/2405.03162

## Visual Language Intelligence and Edge AI 2.0 VILA 1.5
https://developer.nvidia.com/.../visual-language-models.../

Deploy on Jetson Orin/RTX 4090:
- Paper: https://arxiv.org/abs/2312.07533
- Repo: https://github.com/Efficient-Large-Model/VILA
- HF-repo: https://huggingface.co/Efficient-Large-Model

## TELA: Text to 3D Clothed Humans
GitHub_Link (https://github.com/DongJT1996/TELA)

## Revealing the Parametric Knowledge of Language Models: A Unified Framework for Attribution Methods
https://twitter.com/fly51fly/status/1785423963243647156
https://arxiv.org/abs/2404.18655

## MedSegDiff: Medical Image Segmentation with Diffusion Model
GitHub_Link (https://github.com/KidsWithTokens/MedSegDiff)

## Photoswap : Personalized Subject Swapping in Images
GitHub_Link (https://github.com/eric-ai-lab/photoswap)

## Meta, Llama 3 공개

### Llama 3의 첫 두 가지 모델(사전학습 및 명령어 미세조정된 8B와 70B 모델)을 공개
광범위한 업계 벤치마크들에서 최첨단 성능을 보여주며, 향상된 추론 등 새로운 기능을 제공
현재 사용 가능한 최고의 독점 모델과 동등한 수준의 최고의 오픈 모델을 구축하고자 함. 개발자 피드백을 반영하고, 빠르게 자주 릴리즈하는 것을 목표로 함

### Llama Guard 2, Code Shield, CyberSec Eval 2 등의 새로운 신뢰 및 안전 도구 도입
향후 몇 달 내에 새로운 기능, 더 긴 컨텍스트 윈도우, 추가 모델 크기, 향상된 성능 등을 도입할 예정이며, Llama 3 연구 논문도 공유할 예정
AWS, Databricks, Google Cloud, Hugging Face, Kaggle, IBM WatsonX, Microsoft Azure, NVIDIA NIM, Snowflake 등에서 곧 사용 가능해질 예정이며, AMD, AWS, Dell, Intel, NVIDIA, Qualcomm 등의 하드웨어 플랫폼에서도 지원될 예정
Llama 3 기술로 구축된 Meta AI는 이제 세계 최고 수준의 AI 어시스턴트 중 하나로, 사용자의 지능을 높이고 부담을 덜어줄 수 있음

### Llama 3의 성능
8B와 70B 파라미터 Llama 3 모델은 Llama 2에 비해 큰 도약을 이루었으며, 해당 규모에서 LLM 모델의 새로운 최고 수준을 달성
사전 학습 및 사후 학습의 개선 덕분에 사전 학습되고 명령어 미세 조정된 모델은 8B와 70B 파라미터 규모에서 현존하는 최고의 모델임
사후 학습 절차의 개선으로 거짓 거부율이 상당히 감소하고, 정렬이 개선되었고, 모델 응답의 다양성이 증가함
또한 추론, 코드 생성, 명령어 따르기 등의 기능이 크게 개선되어 Llama 3가 더 조종 가능해짐(Steerable)
Llama 3 개발 과정에서 표준 벤치마크에서의 모델 성능을 살펴보고, 실제 시나리오에 대한 성능 최적화도 추구함
이를 위해 12가지 핵심 사용 사례를 다루는 1,800개의 프롬프트가 포함된 새로운 고품질 인간 평가 세트를 개발함
이 평가 세트를 통해 70B 명령어-추종 모델이 실제 시나리오에서 유사한 크기의 경쟁 모델에 비해 강력한 성능을 보여주는 것으로 나타남
사전 학습된 모델 또한 해당 규모에서 LLM 모델의 새로운 최첨단 기술을 달성
훌륭한 언어 모델을 개발하기 위해서는 혁신, 확장, 단순성 최적화가 중요하다고 믿음
Llama 3 프로젝트 전반에 걸쳐 모델 아키텍처, 사전 학습 데이터, 사전 학습 확장, 명령어 미세 조정의 네 가지 핵심 요소에 초점을 맞추어 이 설계 철학을 채택함

### 모델 아키텍처
Llama 3에서는 비교적 표준적인 디코더 전용 트랜스포머 아키텍처를 선택함
Llama 2와 비교하여 몇 가지 주요 개선 사항이 있음
Llama 3는 언어를 훨씬 더 효율적으로 인코딩하는 128K 토큰의 어휘를 가진 토크나이저를 사용하여 모델 성능을 상당히 개선함
Llama 3 모델의 추론 효율성을 개선하기 위해 8B와 70B 크기 모두에 걸쳐 그룹화된 쿼리 주의(GQA)를 채택함
셀프 어텐션이 문서 경계를 넘지 않도록 마스크를 사용해 8,192개의 토큰 시퀀스로 모델을 훈련

### 학습 데이터
최고의 언어 모델을 학습시키기 위해서는 대규모 고품질 학습 데이터셋의 큐레이션이 가장 중요함
Llama 3는 공개적으로 사용 가능한 소스에서 수집된 15T 이상의 토큰으로 사전 학습됨
학습 데이터셋은 Llama 2에 사용된 것보다 7배 더 크며, 4배 더 많은 코드를 포함함
향후 다국어 사용 사례를 준비하기 위해 Llama 3 사전 학습 데이터셋의 5% 이상이 30개 이상의 언어를 다루는 고품질 비영어 데이터로 구성됨

### 사전 학습 확장
Llama 3 모델에서 사전 학습 데이터를 효과적으로 활용하기 위해 사전 학습 확장에 상당한 노력을 기울임
특히 다운스트림 벤치마크 평가를 위한 일련의 상세한 스케일링 법칙을 개발함
이러한 스케일링 법칙을 통해 최적의 데이터 믹스를 선택하고 학습 컴퓨팅을 최상으로 사용하는 방법에 대해 정보에 입각한 결정을 내릴 수 있음

### 명령어 미세 조정
채팅 사용 사례에서 사전 학습된 모델의 잠재력을 완전히 발휘하기 위해 명령어 조정 접근 방식에 대해서도 혁신을 이룸
사후 학습에 대한 접근 방식은 지도 학습 미세 조정(SFT), 거부 샘플링, 근접 정책 최적화(PPO), 직접 정책 최적화(DPO)의 조합임
SFT에 사용되는 프롬프트의 품질과 PPO 및 DPO에 사용되는 선호도 순위는 정렬된 모델의 성능에 과도한 영향을 미침

### Llama 3로 구축하기
Meta의 비전은 개발자가 Llama 3을 맞춤 설정하여 관련 사용 사례를 지원하고 모범 사례를 쉽게 채택하고 개방형 생태계를 개선할 수 있도록 하는 것임
이번 릴리스에서는 Llama Guard 2 및 Cybersec Eval 2와 함께 업데이트된 구성 요소를 포함한 새로운 신뢰 및 안전 도구와 LLM에서 생성한 안전하지 않은 코드를 필터링하기 위한 추론 시간 가드레일인 Code Shield를 도입함
또한 Llama 3을 LLM을 쉽게 작성, 미세 조정 및 실험할 수 있는 새로운 PyTorch 기본 라이브러리인 torchtune과 함께 개발함
책임감 있는 개발과 배포를 위한 시스템 수준 접근법
Llama 3 모델은 최대한 도움이 되면서도 업계 최고 수준의 책임감 있는 배포 접근 방식을 보장하도록 설계됨
이를 위해 Llama의 책임감 있는 개발과 배포를 위한 새로운 시스템 수준 접근법을 채택함
Llama 모델을 개발자가 고유한 최종 목표를 염두에 두고 설계하는 시스템의 기본 요소로 간주함
명령어 미세 조정은 모델의 안전성을 보장하는 데 중요한 역할을 함
명령어 미세 조정된 모델은 내부 및 외부 노력을 통해 안전성에 대해 레드팀(테스트)을 거침
이러한 노력은 반복적이며 릴리스되는 모델의 안전성 미세 조정에 사용됨
Llama Guard 모델은 프롬프트 및 응답 안전의 기반이 되며 애플리케이션 요구 사항에 따라 새로운 분류를 쉽게 만들 수 있음
새로운 Llama Guard 2는 업계 표준 지원을 위해 최근 발표된 MLCommons 분류법을 사용함
CyberSecEval 2는 LLM의 코드 인터프리터 악용 성향, 공격적인 사이버 보안 기능, 프롬프트 주입 공격에 대한 취약성 측정을 추가하여 이전 버전을 확장함
Code Shield는 LLM에서 생성된 안전하지 않은 코드에 대한 추론 시간 필터링을 지원하여 안전하지 않은 코드 제안, 코드 인터프리터 악용 방지, 보안 명령 실행과 관련된 위험을 완화함

### Llama 3의 대규모 배포
Llama 3는 클라우드 제공업체, 모델 API 제공업체 등 주요 플랫폼에서 곧 사용 가능해질 예정임
벤치마크에 따르면 토크나이저는 Llama 2에 비해 최대 15% 적은 토큰을 생성하여 토큰 효율성이 향상됨
또한 그룹 쿼리 주의력(GQA)이 Llama 3 8B에도 추가됨

### Llama 3의 향후 계획
Llama 3 8B 및 70B 모델은 Llama 3 출시 계획의 시작에 불과함
향후 몇 달 동안 멀티모달, 다국어 대화 능력, 훨씬 더 긴 맥락 창, 전반적으로 더 강력한 기능 등 새로운 기능을 갖춘 여러 모델을 출시할 예정임
Llama 3 학습이 완료되면 상세한 연구 논문도 게재할 예정임

### Llama-3-400B+ will mark the watershed moment that the community gains open-weight access to a GPT-4-class model.
https://github.com/openai/simple-evals

## Garment3DGen : 3D Garment Stylization and Texture Generation (2403, Meta)

site : https://nsarafianos.github.io/garment3dgen
paper : https://arxiv.org/abs/2403.18816
code : comimg 


## NEW EULER SMEA DYN SAMPLER!!! ]

A1111 , ComfyUI 에서 사용할 수 있다고 합니다.

git : https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler

주요내용 >
우수한 이미지를 생성하도록 설계된 오일러의 접근 방식을 기반으로 한 샘플링 방법입니다.
SMEA 샘플러는 대형 이미지를 생성할 때 발생하는 구조적 및 사지 붕괴를 크게 완화할 수 있으며, 상당 부분 우수한 손 묘사를 생성할 수 있습니다(완벽하지는 않지만 기존 샘플링 방법보다 우수함).
SMEA 샘플러는 대부분의 이미지 크기를 수용하도록 설계되었으며 특히 큰 이미지에서 탁월한 성능을 발휘합니다. 또한 훈련 데이터가 충분하지 않은 색다른 크기의 이미지 생성도 지원합니다(예: SDXL에서 512x512 실행, SD1.5에서 823x1216 실행, 640x960 실행 등).
SMEA 샘플러는 SD1.5에서 매우 잘 작동하지만 SDXL에서는 그 효과가 뚜렷하지 않습니다.
계산 리소스 소비 측면에서 Euler dy는 Euler a와 거의 동일하지만 Euler SMEA Dy 샘플러는 약 1.25배 더 많은 계산 리소스를 소비합니다.

This is really good, isn't it? Just using the sampler update, you can get good results at non-standard resolutions with SD15. It's available for use in A1111 and ComfyUI.
git : https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler

A sampling method based on Euler's approach, designed to generate superior imagery.
The SMEA sampler can significantly mitigate the structural and limb collapse that occurs when generating large images, and to a great extent, it can produce superior hand depictions (not perfect, but better than existing sampling methods).
The SMEA sampler is designed to accommodate the majority of image sizes, with particularly outstanding performance on larger images. It also supports the generation of images in unconventional sizes that lack sufficient training data (for example, running 512x512 in SDXL, 823x1216 in SD1.5, as well as 640x960, etc.).
The SMEA sampler performs very well in SD1.5, but the effects are not as pronounced in SDXL.
In terms of computational resource consumption, the Euler dy is approximately equivalent to the Euler a, while the Euler SMEA Dy sampler will consume more computational resources, approximately 1.25 times more.

## gpt-prompt-engineer
https://github.com/mshumer/gpt-prompt-engineer

## WonderJourney: Going from Anywhere to Everywhere
GitHub_Link (https://github.com/KovenYu/WonderJourney)

## StructLDM: Structured Latent Diffusion for 3D Human Generation 
(2404, S-Lab Nanyang Technological University)
site : https://taohuumd.github.io/projects/StructLDM/
paper : https://arxiv.org/abs/2404.01241

## Llama2, Mistral 모델의 FP8
Friendli AI 에서 Llama2, Mistral 모델의 FP8 모델 체크포인트를 공개했습니다. 
[https://huggingface.co/FriendliAI/Mistral-7B-Instruct-v0.2-fp8](https://huggingface.co/FriendliAI/Mistral-7B-Instruct-v0.2-fp8)
[https://huggingface.co/FriendliAI/Llama-2-7b-chat-hf-fp8](https://huggingface.co/FriendliAI/Llama-2-7b-chat-hf-fp8)
[https://huggingface.co/FriendliAI/Llama-2-13b-chat-hf-fp8](https://huggingface.co/FriendliAI/Llama-2-13b-chat-hf-fp8)
[https://huggingface.co/FriendliAI/Llama-2-70b-chat-hf-fp8](https://huggingface.co/FriendliAI/Llama-2-70b-chat-hf-fp8)

## Retrieval-based-Voice-Conversion
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main

자기들 목소리 anime girl로 바꾸고 싶어서 만든것.

## GPT beats diffusion
GitHub_Link (https://github.com/FoundationVision/VAR)

## InstantID : Zero-shot Identity-Preserving Generation in Seconds
gitHub_Link (https://github.com/InstantID/InstantID)

## InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction
https://twitter.com/arankomats.../status/1774618885494342119
https://arxiv.org/abs/2403.19652

## Google announces Streaming Dense Video Captioning
https://twitter.com/_akhaliq/status/1775176791772008825
https://huggingface.co/papers/2404.01297

## Large Language Models Are Effective Temporal Learners
https://twitter.com/_akhaliq/status/1775179607920017806
https://huggingface.co/papers/2404.00308

## ComfyUI
Node : https://github.com/chaojie/ComfyUI-AniPortrait 

## Tokenizer Choice For LLM Training: Negligible or Crucial?
https://arxiv.org/pdf/2310.08754.pdf
- 단일 언어 토크나이저를 기반으로 개발된 LLM의  다국어 성능이 비교적 낮은 점, 코딩 특화 토크나이저를 이용한 LLM의 코딩 능력을 개선한 점 등의 사례를 통해 토크나이저가 LLM의 성능에 큰 영향을 미친다는 것이 다시금 확인 되었습니다.
- 토크나이저의 vocab size는 무작정 늘리는 것이 좋다기보단 추론 속도와 메모리 사용량을 종합적으로 고려해 최적의 값을 찾는 게 중요합니다.
- 토크나이저 자체를 평가하는 지표(fertility, parity)와 LLM의 성능 지표 간에 강한 관계는 없는 것으로 나타났습니다.
- 50B 이상의 모델들을 파인튜닝할 때는 토크나이저를 바꾸는 것이 LLM의 성능에 영향을 주지 않았다고 합니다.
(Abstract translated with Claude Opus)
- 토큰화는 현대 LLM의 과소 연구되고 종종 간과되는 구성 요소입니다. 대부분의 발표된 연구는 토큰화를 최적화하기 위한 절제(ablation)나 분석을 수행하지 않고, 종종 다른 모델에서 차용한 단일 토크나이저를 모든 실험에 사용합니다. 또한, 기본 모델을 fine-tuning할 때 토크나이저는 일반적으로 변경되지 않은 상태로 유지됩니다. 
- 이 논문에서는 토크나이저의 크기, 사전 토큰화 정규 표현식 및 학습 데이터가 모델의 생성 속도, 유효 컨텍스트 크기, 메모리 사용량 및 다운스트림 성능에 상당한 영향을 미칠 수 있음을 보여줍니다. 
- 우리는 전문화된 Byte-Pair Encoding 코드 토크나이저를 학습시키고, HumanEval 및 MBPP와 같은 코드 생성 작업에 대한 LLM의 성능에 미치는 토크나이저 설계의 영향에 대해 광범위한 절제(ablation)를 수행하며, 토크나이저 하이퍼 파라미터 선택 및 사전 학습된 LLM에서의 토크나이저 전환에 대한 권장 사항을 제공합니다. 
- 우리는 처음부터 학습한 모델과 사전 학습된 모델에서 실험을 수행하여 광범위한 사용 사례에 대한 적용 가능성을 검증합니다. 우리는 500억 개 이상의 토큰으로 fine-tuning할 때, 사전 학습된 LLM의 토크나이저를 전문화하여 생성 속도와 유효 컨텍스트 크기에서 큰 이득을 얻을 수 있다는 것을 발견했습니다.

## Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation
https://twitter.com/fly51fly/status/1773835840709697889
https://arxiv.org/abs/2403.19319

## 대규모 언어 모델(LLM) 격투장
스트리트 파이터 III에서 LLM이 실시간으로 서로 싸우게 했답니다.
어떤 LLM이 최고의 파이터가 될까요?
깃허브 https://github.com/OpenGenerativeAI/llm-colosseum

## Boosting LLMs with Novel Iterative Data Enhancement
https://huggingface.co/papers/2403.15042

## DE-Net: Dynamic Text-guided Image Editing Adversarial Networks
GitHub_Link (https://github.com/tobran/DE-Net)

## AutoRecon: Automated 3D Object Discovery and Reconstruction
GitHub_Link (https://github.com/zju3dv/AutoRecon)
## Mixing Expert LLMs into a Mixture-of-Experts LLM
https://huggingface.co/papers/2403.07816

## Visual Style Prompting with Swapping Self-Attention
Official Pytorch implementation of "Visual Style Prompting with Swapping Self-Attention"
GitHub_Link (https://github.com/naver-ai/Visual-Style-Prompting)

## Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU
https://twitter.com/_akhaliq/status/1767393991727657262
https://huggingface.co/papers/2403.06504

## Domain Expansion of Image Generators
Domain Expansion of Image Generators offers a groundbreaking perspective on enhancing pretrained generative models. The ability to seamlessly integrate numerous new domains while preserving the original knowledge presents a transformative approach to model versatility and efficiency, potentially reshaping the landscape of generative model applications.
GitHub_Link (https://github.com/adobe-research/domain-expansion)

## DreamCraft3D
GitHub_Link (https://github.com/deepseek-ai/DreamCraft3D)

DreamCraft3D pioneers a groundbreaking approach to 3D content generation, overcoming consistency challenges with innovative techniques like score distillation and Bootstrapped Score Distillation. The alternating optimization strategy showcases a synergistic relationship between 3D scene representation and diffusion models, resulting in remarkable photorealistic renderings and a noteworthy leap in the state-of-the-art.

## Intel Extension for Transformers
Intel Extension for Transformers (https://github.com/intel/intel-extension-for-transformers) supports INT4 and low-bit inference on both CPUs and GPUs!
📔Simple usage guide: https://github.com/intel/intel-extension-for-transformers/blob/main/docs/weightonlyquant.md 
🔥All your need is to get an Intel GPU and run LLMs @huggingface
 
 https://github.com/intel/intel-extension-for-transformers

## Byte-gpt 
https://byte-gpt.github.io/
https://arxiv.org/abs/2402.19155
byte level transformer (bGPT)
tokenize 없이 byte 정보를 바로 모델이 넣고 prediction 하는 방식으로 동작하는 모델.. 그래서 vocab size는 256 + 1.
공개된 코드 상으로는 sequence length는 512에 불과하지만 multimodal로 가는 과정

## Keyframer (Apple) 
 LLM 의 코드 생성 기능을 활용하여 SVG 벡터 이미지를 코드로 변경하여 애니메이션화. 텍스트 프롬프트를 입력으로 영상을 생성하는 기존 생성 모델과는 다르게, 자연어와 이미지(SVG) 을 넣어주면 LLM 코드 생성 기능 활용하여 애니메이트 생성 

Keyframer: Empowering Animation Design using Large Language Models (2402, Apple)

paper : https://arxiv.org/abs/2402.06071

 대규모 언어 모델(LLM)은 다양한 크리에이티브 영역에 영향을 미칠 수 있는 잠재력을 가지고 있지만, 애니메이션에 LLM을 적용하는 것은 잘 알려지지 않았으며 사용자가 자연어로 동작을 효과적으로 설명하는 방법과 같은 새로운 과제를 제시합니다. 이 논문에서는 정적 이미지(SVG)를 자연어로 애니메이션화하는 디자인 도구인 Keyframer를 소개합니다. 전문 애니메이션 디자이너 및 엔지니어와의 인터뷰를 통해 얻은 정보를 바탕으로 Keyframer는 생성된 결과물의 프롬프트와 직접 편집을 결합하여 애니메이션을 탐색하고 다듬을 수 있도록 지원합니다. 또한 사용자가 디자인 변형을 요청할 수 있어 비교와 아이디어 도출을 지원합니다. 13명의 참가자를 대상으로 한 사용자 연구를 통해 모션을 설명하는 의미론적 프롬프트 유형 분류와 사용자가 생성된 출력에 따라 지속적으로 목표를 조정하는 '분해된' 프롬프트 스타일을 포함한 사용자 프롬프트 전략의 특성을 분석하고, 프롬프트와 함께 직접 편집을 통해 오늘날 생성 도구에서 흔히 사용되는 단발성 프롬프트 인터페이스 이상의 반복을 가능하게 하는 방법을 공유합니다. 이 작업을 통해 다양한 시청자가 애니메이션 제작에 참여할 수 있도록 LLM을 활용하는 방법을 제안합니다.

 
##  Audio2Video 모델인 EMO 결과물 공개
https://humanaigc.github.io/emote-portrait-alive
이미지와 음성 오디오를 이용하여 영상을 생성하는 모델의 결과물을 공개했습니다. 
결과를 보면 상당히 퀄리티가 좋은 것을 볼 수 있는데요. 
이미지 한 장으로 이런 결과가 나온다는게 신기하네요.

##  nvnv-bianca
3억토큰으로 학습시킨 한국어 소설 AI입니다.
https://huggingface.co/instructkr/nvnv-bianca
10.8b 야놀자의 EEVE 모델을 기반으로 제작되었으며, 8k 컨텍스트를 지원합니다.
여러 번 사용 해본 결과 맥락을 잘 잇고 묘사 능력이 뛰어납니다.
eos 토큰 비활성화 시킨 후 사용해야 더욱 편리합니다.

## Phind-70B 공개 - GPT-4 Turbo와 코드 품질 격차를 줄이면서 4배 빠른 실행 가능한 모델 
- 초당 최대 80개의 토큰을 처리(GPT-4 Turbo는 초당 ~20토큰)
32K 토큰 윈도우를 지원
CodeLlama-70B 모델 기반으로 추가적인 50B 토큰으로 파인튜닝됨
HumanEval 에서 82.3%를 기록해서 81%인 GPT-4 Turbo(gpt-3-0125-preview)를 상회함
Meta의 CRUXEval 에서는 59%로 GPT-4의 62%에 조금 못 미침
코드생성 측면에서는 거의 GPT-4 Turbo와 동일하거나 일부 작업에서는 이를 능가
GPT-4 Turbo 보다 덜 "Lazy" 해서 상세한 코드 예제를 생성하는데 주저하지 않음
phind.com
## Lumiere is a space-time diffusion research model 
generates video from various inputs, including image-to-video. The model generates videos that start with the desired first frame & exhibit intricate coherent motion across the entire video duration.
Website: https://lumiere-video.github.io/
Paper: https://arxiv.org/abs/2401.12945
YouTube: https://www.youtube.com/watch?v=wxLr02Dz2Sc

## STMC can generate 3D human motion from text with multi-track timeline control!
https://twitter.com/dreamingtulpa/status/1749778661517959324
https://mathis.petrovich.fr/stmc/?ref=aiartweekly

## The usual RAG approach struggles with retrieval accuracy when faced with massive indistinguishable documents comprising text, tables, and images.
- https://arxiv.org/pdf/2402.01767.pdf
- https://github.com/TebooNok/HiQA

## MetaVoice-1B
MetaVoice-1B is a 1.2B parameter base model trained on 100K hours of speech for TTS (text-to-speech). It has been built with the following priorities:
https://github.com/metavoiceio/metavoice-src 

## CodeLlama-70B PostgreSQL、SQLCoder-70B。
https://huggingface.co/defog/sqlcoder-70b-alpha

## Argmax presents WhisperKit
https://huggingface.co/argmaxinc/whisperkit-coreml

## Awesome-Graph-LLM
그래프 기반 기법과 LLM과 관련된 연구 논문의 큐레이션 목록 저장소.
https://github.com/XiaoxinHe/Awesome-Graph-LLM

## codelaama2
quantized CodeLlama 70b Instruct to 4-bit with MLX
[https://huggingface.co/.../CodeLlama-70b-Instruct-hf-4bit...
](https://huggingface.co/mlx-community/CodeLlama-70b-Instruct-hf-4bit-MLX?fbclid=IwAR0KYHovfFxB87OvVg55RLkIre4N0JfQbi0fPYbemZDJQ3K8Ka-fZnKM4sA)
## Towards Conversational Diagnostic AI

의료의 핵심은 의사와 환자 간의 대화이며, 숙련된 병력 청취는 정확한 진단, 효과적인 관리, 지속적인 신뢰의 토대가 됩니다. 
진단 대화를 할 수 있는 인공지능(AI) 시스템은 접근성, 일관성, 치료의 질을 높일 수 있습니다. 
하지만 임상의의 전문 지식에 근접하는 것은 매우 어려운 과제입니다. 
여기에서는 진단 대화에 최적화된 대규모 언어 모델(LLM) 기반 AI 시스템인 AMIE(Articulate Medical Intelligence Explorer)를 소개합니다. 
AMIE는 다양한 질병 상태, 전문 분야 및 상황에 따라 학습을 확장하기 위해 자동화된 피드백 메커니즘을 갖춘 새로운 셀프 플레이 기반 시뮬레이션 환경을 사용합니다. 
병력 청취, 진단 정확도, 관리 추론, 의사소통 기술, 공감 능력 등 임상적으로 의미 있는 성과 축을 평가하기 위한 프레임워크를 설계했습니다. 
객관적 구조화 임상시험(OSCE) 방식으로 검증된 환자 행위자와의 텍스트 기반 상담에 대한 무작위 이중맹검 교차 연구를 통해 AMIE의 성과를 1차 진료 의사(PCP)의 성과와 비교했습니다.
이 연구에는 캐나다, 영국, 인도의 임상 제공자가 제공한 149개의 사례 시나리오, AMIE와 비교하기 위한 20개의 PCP, 전문 의사와 환자 배우의 평가가 포함되었습니다.
전문 의사가 평가한 32개 축 중 28개 축, 환자 행위자가 평가한 26개 축 중 24개 축에서 AMIE가 더 높은 진단 정확도와 우수한 성능을 보여주었습니다.
이번 연구에는 몇 가지 한계가 있으므로 적절한 주의를 기울여 해석해야 합니다. 
임상의들은 익숙하지 않은 동기식 텍스트 채팅으로 제한되었으며, 이는 대규모의 LLM-환자 상호작용을 허용하지만 일반적인 임상 실습을 대표하지 않습니다. 
AMIE를 실제 환경에 적용하기 위해서는 더 많은 연구가 필요하지만, 이번 연구 결과는 대화형 진단 AI를 향한 이정표가 될 것입니다.
- Blog: https://blog.research.google/2024/01/amie-research-ai-system-for-diagnostic_12.html
- arXiv: https://arxiv.org/abs/2401.05654
- Browse: https://browse.arxiv.org/pdf/2401.05654.pdf
- PDF: https://arxiv.org/pdf/2401.05654.pdf  
- arXiv-vanity: https://www.arxiv-vanity.com/papers/2401.05654 
- Paper page: https://huggingface.co/papers/2401.05654 
- HTML : https://browse.arxiv.org/html/2401.05654v1 
- Papers with code: https://paperswithcode.com/paper/towards-conversational-diagnostic-ai

## Introducing DeepSeekMoE
https://github.com/deepseek-ai/DeepSeek-MoE

## Transformers are Multi-State RNNs
논문 초록
트랜스포머는 이전 세대의 최첨단 자연어 처리 모델인 순환 신경망(RNN)과 비교했을 때 개념적으로 다른 것으로 간주됩니다. 
이 연구에서는 디코더 전용 트랜스포머가 실제로 무한 다중 상태 RNN, 즉 숨겨진 상태 크기가 무제한인 RNN 변형으로 개념화될 수 있음을 보여줍니다. 
또한 숨겨진 상태의 크기를 고정하여 사전 훈련된 트랜스포머를 유한 다중 상태 RNN으로 변환할 수 있음을 보여줍니다. 
기존의 여러 트랜스포머 캐시 압축 기법이 이러한 변환 정책으로 구성될 수 있음을 관찰하고, 이러한 정책과 비교하여 더 간단한 새로운 정책인 TOVA를 소개합니다. 
몇 가지 장거리 작업에 대한 실험 결과, TOVA는 다른 모든 기본 정책보다 성능이 뛰어나면서도 전체(무한) 모델과 거의 동등하고 경우에 따라서는 원래 캐시 크기의 18개만 사용하는 것으로 나타났습니다. 
연구 결과에 따르면 트랜스포머 디코더 LLM이 실제로는 RNN처럼 작동하는 경우가 많습니다. 
또한 가장 골치 아픈 계산 병목 현상 중 하나인 캐시 메모리 크기를 완화할 수 있는 옵션도 제시합니다.
논문 https://arxiv.org/abs/2401.06104

## 업스테이지의 솔라 논문
32층 기본 모델에서 시작하여, 이 모델의 일부를 복제하여 연결함으로써 48층의 확장 모델을 생성하는 방식 (Depth Up-Scaling 라고 명명)

[모델 구조]
1. 32 layer Llama 2 architecture with Mistral 7B pretrained weights
2. 단순 복제하여 2개 세트 생성
3. 첫번째 세트의 끝 8 layer, 두번째 세트의 처음 8 layer를 잘라냄 -> 24 layer * 2 model
4. 합쳐서 48 layer (10.7 billion parameters)

[학습 방법]
1. Instruction Tuning: QA 포맷 학습 (오픈소스 + 합성 math QA 데이터)
2. Alignment Tuning: DPO 기반 튜닝 ( {prompt, chosen, rejected} tuple로 만들어서 DPO 진행)
3. Model Merging: 단순 weight 평균과 SLERP 활용
 
## Starling-7b-alpha (@BanghuaZet al.) is a new 7B LLM 
uses a brand-new reward model and policy optimization method. 
it approaches GPT-4 in perf on MT Bench, MMLU, and more (beating Claude, 3.5, etc.)
https://twitter.com/jerryjliu0/status/1735842203241759099
[https://docs.llamaindex.ai/models/llms.html
](https://docs.llamaindex.ai/en/latest/module_guides/models/llms.html?fbclid=IwAR3QgJYKlQ7cINcA5xHh4P9fue8jNPEs1f0w99ocizRV2bLupnCDn-zS4lA#open-source-llms)

## Local RAG on Windows
https://twitter.com/llama_index/status/1736429047956349058
https://github.com/marklysze/LlamaIndex-RAG-WSL-CUDA

## Towards LangChain 0.1: LangChain-Core and LangChain-Community
https://twitter.com/LangChainAI/status/1734641665556857148
[https://blog.langchain.dev/the-new-langchain.../
](https://blog.langchain.dev/the-new-langchain-architecture-langchain-core-v0-1-langchain-community-and-a-path-to-langchain-v0-1/?fbclid=IwAR2DDz1sTB97NaUt2PX-0nJuDG4g0oQ6LdBbJeww4BN8QRc51uSCmCyMvCY)

## MLC LLM

Documentation | Blog | Discord

Machine Learning Compilation for Large Language Models (MLC LLM) is a high-performance universal deployment solution that allows native deployment of any l
arge language models with native APIs with compiler acceleration. The mission of this project is to enable everyone to develop, optimize and deploy AI models natively on everyone's devices with ML compilation techniques.

https://github.com/mlc-ai/mlc-llm

## Pipegoose: end-to-end framework for training multi-modal MoE in a decentralized way. (DiLoCo)[https://arxiv.org/abs/2311.08105]를 replicate하는 오픈소스 프로젝트
Repo: https://github.com/xrsrke/pipegoose

## MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers

드디어 나올 게 나오는군요.

3D 메쉬를 생성하는 모델입니다. GPT처럼 Transformer 의 디코드 부분만 사용했답니다.

"MeshGPT는 학습된 기하학적 어휘에서 토큰을 생성하도록 훈련된 트랜스포머 모델에서 자동회귀적으로 샘플링하여 삼각형 메시를 생성합니다. 그런 다음 이러한 토큰을 트라이앵글 메시의 면으로 디코딩할 수 있습니다. 이 방법은 선명한 모서리와 높은 충실도가 특징인 깨끗하고 일관성 있으며 컴팩트한 메시를 생성합니다."

프로젝트 https://nihalsid.github.io/mesh-gpt/
## Gemini 

Welcome to the Gemini era
https://deepmind.google/technologies/gemini
Introducing Gemini: our largest and most capable AI model
https://blog.google/technology/ai/google-gemini-ai/
Enabling next-generation AI workloads: Announcing TPU v5p and AI Hypercomputer
https://cloud.google.com/.../introducing-cloud-tpu-v5p...
1. Testing Gemini: Finding connections: https://www.youtube.com/watch?v=Rn30RMhEBTs
2. Hands-on with Gemini: Interacting with multimodal AI: https://www.youtube.com/watch?v=UIZAiXYceBI
3. Gemini: Google’s newest and most capable AI model: https://www.youtube.com/watch?v=jV1vkHv4zq8
4. Testing Gemini: Turning images into code: https://www.youtube.com/watch?v=NHLnjWTEZps
5. Testing Gemini: Emoji Kitchen: https://www.youtube.com/watch?v=ki8kRJPXCW0
6. Gemini: All you need to know in 90 seconds: https://www.youtube.com/watch?v=_TVnM9dmUSk
7. Testing Gemini: Understanding environments: https://www.youtube.com/watch?v=JPwU1FNhMOA
8. Gemini: Explaining reasoning in math and physics: https://www.youtube.com/watch?v=K4pX1VAxaAI
9. Gemini: Excelling at competitive programming: https://www.youtube.com/watch?v=LvGmVmHv69s
10. Testing Gemini: Fit check: https://www.youtube.com/watch?v=HP2pNdCRT5M
11. Gemini: Processing and understanding raw audio: https://www.youtube.com/watch?v=D64QD7Swr3s
12. Testing Gemini: Guess the movie: https://www.youtube.com/watch?v=aRyuMNwn02w
13. Mark Rober takes Bard with Gemini Pro for a test flight: https://www.youtube.com/watch?v=mHZSrtl4zX0
14. Gemini: Safety and responsibility at the core: https://www.youtube.com/watch?v=gi6J_WjjNhE
15. Gemini: Reasoning about user intent to generate bespoke experiences: https://www.youtube.com/watch?v=v5tRc_5-8G4
16. Gemini: Unlocking insights in scientific literature: https://www.youtube.com/watch?v=sPiOP_CB54A
17. Using AI to Improve Students writing skills Quill.org x Google.org: https://www.youtube.com/watch?v=f0pMe4aFXx0

## Information Retrieval:  Who wins, GPT-4-Turbo or a RAG based on GPT4?
https://github.com/A-Roucher/LLM_vs_RAG_NeedleInAHaystack

## MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers

드디어 나올 게 나오는군요.

3D 메쉬를 생성하는 모델입니다. GPT처럼 Transformer 의 디코드 부분만 사용했답니다.

"MeshGPT는 학습된 기하학적 어휘에서 토큰을 생성하도록 훈련된 트랜스포머 모델에서 자동회귀적으로 샘플링하여 삼각형 메시를 생성합니다. 그런 다음 이러한 토큰을 트라이앵글 메시의 면으로 디코딩할 수 있습니다. 이 방법은 선명한 모서리와 높은 충실도가 특징인 깨끗하고 일관성 있으며 컴팩트한 메시를 생성합니다."

프로젝트 https://nihalsid.github.io/mesh-gpt/

## Can GPT-4V(ision) Serve Medical Applications? Case Studies on GPT-4V for Multimodal Medical Diagnosis
요약: 
대규모 기반 모델에 힘입어 인공지능의 개발은 최근 엄청난 진전을 이루었으며, 대중의 관심이 급증하고 있습니다. 이 연구에서는 특히 멀티모달 의료 진단 영역에서 OpenAI의 최신 모델인 GPT-4V(ision)의 성능을 평가하는 것을 목표로 합니다. 평가 대상은 중추신경계, 두경부, 심장, 흉부, 혈액학, 간담도, 위장, 비뇨생식기, 부인과, 산부인과, 유방, 근골격계, 척추, 혈관, 종양학, 외상, 소아과 등 17개 인체 시스템으로, 일상적인 임상에서 사용되는 8개 모달리티에서 촬영한 이미지를 포함합니다, 엑스레이, 컴퓨터 단층 촬영(CT), 자기 공명 영상(MRI), 양전자 방출 단층 촬영(PET), 디지털 감산 혈관 조영술(DSA), 유방 조영술, 초음파 및 병리학. 우리는 영상 양식 및 해부학 인식, 질병 진단, 보고서 생성, 질병 위치 파악 등 다양한 임상 작업에서 특허 이력 제공 여부에 관계없이 GPT-4V의 능력을 조사했습니다.
연구 결과에 따르면 GPT-4V는 의료 영상 양식과 해부학을 구분하는 데는 능숙하지만 질병 진단과 종합적인 보고서 생성에는 상당한 어려움을 겪고 있는 것으로 나타났습니다. 이러한 결과는 대규모 멀티모달 모델이 컴퓨터 비전과 자연어 처리 분야에서 상당한 발전을 이루었지만, 실제 의료 애플리케이션과 임상 의사 결정을 효과적으로 지원하는 데는 아직 멀었다는 점을 강조합니다.
arXiv: https://arxiv.org/abs/2310.09909
Browse: https://browse.arxiv.org/pdf/2310.09909.pdf
PDF: https://arxiv.org/pdf/2310.09909.pdf  
Paper page: https://huggingface.co/papers/2310.09909 
Papers with code: https://huggingface.co/papers/2310.09909
GitHub: https://github.com/chaoyi-wu/GPT-4V_Medical_Evaluation

##  A Survey of Large Language Models in Medicine: Progress, Application, and Challenge
요약: 
ChatGPT와 같은 대규모 언어 모델(LLM)은 인상적인 인간 언어 이해 및 생성 능력으로 인해 상당한 주목을 받고 있습니다. 따라서 의사와 환자 치료를 지원하기 위해 의료 분야에서 LLM을 적용하는 것은 인공지능과 임상의학 모두에서 유망한 연구 방향으로 부상하고 있습니다. 이를 위해 본 조사 연구에서는 의학 분야에서의 인공신경망의 현재 진행 상황, 응용 분야, 직면한 과제에 대한 포괄적인 개요를 제공합니다. 특히 다음과 같은 질문을 다루고자 합니다: 1) LLM이란 무엇이며 의료용 LLM은 어떻게 구축할 수 있는가? 2) 의료용 LLM의 다운스트림 성과는 무엇인가요? 3) 의료용 LLM은 실제 임상에서 어떻게 활용될 수 있나요? 4) 의료용 LLM을 사용할 때 어떤 문제가 발생하나요? 5) 어떻게 하면 의료용 LLM을 더 잘 구축하고 활용할 수 있을까요? 결과적으로 이 조사 연구는 의학 분야에서 LLM의 기회와 도전 과제에 대한 인사이트를 제공하고 실용적이고 효과적인 의학 LLM을 구축하기 위한 귀중한 리소스로 활용되는 것을 목표로 합니다. 정기적으로 업데이트되는 의료 LLM의 실용적인 가이드 리소스 목록은 다음 https URL에서 확인할 수 있습니다.

arXiv: https://arxiv.org/abs/2311.05112
Browse: https://browse.arxiv.org/pdf/2311.05112.pdf
PDF: https://arxiv.org/pdf/2311.05112.pdf  

## KoLLM-LogBook 
 큰 주목을 받고 있는 "OpenHermes-2-Mistral-7B" 모델을 개발한 teknium의 "LLM-Logbook" 프로젝트의 한국어 버전입니다.
 주요 목표는 Multiple-Choice Question Answering (MCQA)를 기반으로 한 언어모델 평가 방법론을 넘어서, 다양한 프롬프트에 대한 언어모델의 생성결과를 직접 비교하는 것으로 총 100개의 프롬프트와 서로 다른 언어 모델들의 답변을 기록하고 있습니다.
KoLLM-LogBook에는 금융, 수학, 의학, 프로그래밍, 창작 글쓰기 등 총 15개 분야에서 제작한 100개의 프롬프트와 다양한 언어 모델들의 응답이 수록되어 있습니다. 
현재 프로젝트에는 다음 4개의 모델 결과가 포함
amphora/small-instruct
kyujinpy/KoR-Orca-Platypus-13B
krevas/LDCC-Instruct-Llama-2-ko-13B-v4
gpt-3.5-turbo-0613
Compare Models 페이지 에서는 동일 프롬프트에 대한 서로 다른 모델의 답변을 비교하실 수 있고
Model Reports 페이지에서는 프롬프트 전체에 대한 개별 모델의 답변을 모아보실 수 있습니다.
추가하고 싶으신 모델이 있으시거나 궁금하신 점이 있으시면 편하게 연락 부탁드립니다.
github: https://github.com/guijinSON/KoLLM-LogBook/tree/main
streamlit: https://kollm-logbook-qqw6uzf89xizxjilkihjsh.streamlit.app/

## 해리포터가 누구? MS, AI 학습 데이터 중 특정 정보 삭제 기술 공개
저작권 문제 등에 큰 해결책 될 것
인공지능(AI)이 학습한 데이터 중 문제가 있는 일부분만 삭제할 수 있는 기술 공개. 데이터 저작권 문제로 골머리를 앓는 빅테크에 돌파구가 될 수 있다는 분석.
벤처비트는 마이크로소프트(MS) 연구진이 대형언어모델(LLM)에서 특정 정보를 삭제하는 방법을 온라인 논문 사이트 아카이브(arXiv)에 게재했다고 소개
MS 애저 연구원은 메타의 오픈 소스 LLM '라마 2 7B' 모델에 포함된 해리포터에 대한 모든 지식을 삭제하는 데 성공. 논문의 제목도 '해리 포터가 누구? 대략적인  LLM의 학습 취소법(Who’s Harry Potter? Approximate Unlearning in LLMs)'

논문 : https://arxiv.org/abs/2310.02238

## LAVIE: HIGH-QUALITY VIDEO GENERATION WITH CASCADED LATENT DIFFUSION MODELS (2309, Shanghai AI Lab, Nanyang Technological University 외)

고품질 텍스트-비디오(T2V, Text-to-Video) 생성 모델를 기본 T2V 모델, 시간 보간 모델, 비디오 초고해상도 모델로 구성된 계단식 비디오 잠복 확산 모델에서 작동하는 통합 비디오 생성 프레임워크 제안, 실험을 통해 LaVie가 양적, 질적으로 최첨단 성능을 달성

project : [https://vchitect.github.io/LaVie-project/](https://vchitect.github.io/LaVie-project/?fbclid=IwAR2_AuNSz7ZqIklzCoNqoS1J2mWqf-E8q3Ox4ybcVyEtZGuVh3EiNOYKPnk)

## 2.6조 토큰으로 훈련된 130억 매개변수를 가진 다국어 모델 'Baichuan 2'
Baichuan 2: Open Large-scale Language Models : https://arxiv.org/abs/2309.10305v2 , Baichuan

```
본 연구에서는 다국어를 지원하는 대규모 언어 모델 Baichuan 2를 소개합니다. Baichuan 2는 7B(70억 매개변수)와 13B(130억 매개변수)의 두 가지 모델을 보유하고 있으며, 전례 없는 규모인 2.6조 토큰에 의해 훈련되고 있다.
이 대량의 훈련 데이터 덕분에 Baichuan 2는 일반적인 벤치마크 테스트로 이전 버전인 Baichuan 1보다 약 30% 높은 성능을 발휘합니다.
특히 Baichuan 2는 수학 및 프로그래밍 문제에서도 높은 성능을 보이며 의료 및 법률과 같은 전문 영역에서도 우수한 성적을 달성하고 있습니다. Baichuan 2-7B-Chat과 Baichuan 2-13B-Chat이라는 인간의 지시에 따라 최적화된 채팅 모델도 공개되었습니다. 이러한 모델은 상호 작용과 컨텍스트 이해에 특히 우수합니다.
이 모델은 공개된 벤치마크 테스트(MMLU, CMMLU, GSM8K, HumanEval 등)에서 같은 규모의 다른 오픈 소스 모델과 비교하여 동등하거나 그 이상의 성능을 보여줍니다. 또한 의료 및 법률 등 전문 분야에서도 높은 성적을 올리고 있습니다.
이 평가 결과에서 Baichuan 2는 다국어 지원이지만 높은 성능과 광범위한 적용 가능성을 가지고 있음을 확인할 수 있습니다.
```

## Dense Text-to-Image Generation with Attention Modulation (2308, naver)
네이버, 이미지 레이아웃을 만들고 각 해당 영역에 텍스트 프롬프트를 표시하여 이미지 생성
 - (예 background: beach, blue skt,  segment1: girl, segment2: chair)
 - 논문 : https://arxiv.org/abs/2308.12964
 - 소스 : https://github.com/naver-ai/DenseDiffusion
 - 내용:번역
```
기존의 텍스트-이미지 확산 모델은 각 텍스트 프롬프트가 특정 이미지 영역에 대한 자세한 설명을 제공하는 고밀도 캡션이 주어지면 사실적인 이미지를 합성하는 데 어려움을 겪습니다.
이러한 문제를 해결하기 위해 우리는 사전 학습된 텍스트-이미지 확산 모델을 조정하여 장면 레이아웃을 제어하면서 이러한 고밀도 캡션을 처리할 수 있는 학습이 필요 없는 방법인 DenseDiffusion을 제안합니다.
먼저 생성된 이미지의 레이아웃과 사전 학습된 모델의 중간 주의도 맵 간의 관계를 분석합니다.
그런 다음 레이아웃 안내에 따라 특정 영역에 객체가 나타나도록 안내하는 주의 변조 방법을 개발합니다.
추가적인 미세 조정이나 데이터 세트 없이도 자동 및 인간 평가 점수 모두에 대해 조밀한 캡션이 주어졌을 때 이미지 생성 성능을 개선합니다. 또한 레이아웃 조건에 따라 특별히 학습된 모델을 사용하여 유사한 품질의 시각적 결과를 얻을 수 있습니다.
```

## VideoComposer: Compositional Video Synthesis with Motion Controllability (2306, Alibaba/Ant) 
중국 알리바바(modelscope)  Image to video , video to video 
논문 : https://arxiv.org/abs/2306.02018
사이트 : https://modelscope.cn/models/damo/Image-to-Video/summary
       https://modelscope.cn/models/damo/Video-to-Video/summary
```
(내용:번역) 시각적 콘텐츠 제작의 높은 기준으로 제어 가능성을 추구하면서 맞춤형 이미지 합성 분야에서 괄목할 만한 발전이 이루어졌습니다. 그러나 제어 가능한 비디오 합성을 달성하는 것은 시간적 역동성의 큰 변화와 프레임 간 시간적 일관성의 요구 사항으로 인해 여전히 어려운 과제입니다. 본 연구에서는 합성 생성 패러다임에 기반하여 텍스트 조건, 공간 조건, 더 나아가 시간적 조건에 따라 유연하게 영상을 합성할 수 있는 비디오 컴포저(VideoComposer)를 제시합니다. 특히, 비디오 데이터의 특성을 고려하여 압축된 비디오의 모션 벡터를 명시적 제어 신호로 도입하여 시간적 동역학에 대한 가이드를 제공합니다. 또한 순차적 입력의 공간적, 시간적 관계를 효과적으로 통합하기 위한 통합 인터페이스 역할을 하는 시공간 조건 인코더(STC-encoder)를 개발하여 모델이 시간적 조건을 더 잘 활용하고 프레임 간 일관성을 높일 수 있도록 합니다. 광범위한 실험 결과에 따르면 VideoComposer는 텍스트 설명, 스케치 시퀀스, 참조 비디오 또는 단순히 손으로 만든 모션과 같은 다양한 형태로 합성된 비디오 내에서 공간 및 시간 패턴을 동시에 제어할 수 있는 것으로 나타났습니다.
```

*이제 중국 알리바바는 text to image Zeroscope 에 이어 image to video, video to video 모두 갖추게 되었네요

## Meta, SeamlessM4T : 최초, 올인원, 음성/텍스트 multimodal 번역 모델
최첨단 최고(state-of-the-art) 결과물로 이전 시스템의 한계를 극복하는 음성/텍스트 번역 및 트랜스크립션 모델.
여러 언어간 교차 speech-to-text, speech-to-speech, text-to-speech, text-to-text, and speech recognition.  (예:영어 음성을 러시아 텍스트로, 영어 음성을 러시아 음성으로, 영어 텍스트를 러시아 음성으로, 영어 텍스트를 러시아 텍스트로...).
OpenAI의 Whisper 성능을 앞서 최첨단 최고(state-of-the-art) 달성.

- 101 languages for speech input.
- 96 Languages for text input/output.
- 35 languages for speech output.

소개 : https://ai.meta.com/blog/seamless-m4t/
      https://ai.meta.com/resources/models-and-libraries/seamless-communication/
  
논문 : https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf
데모 : https://seamless.metademolab.com/
      https://huggingface.co/spaces/facebook/seamless_m4t 
github : https://github.com/facebookresearch/seamless_communication

## 엔씨소프트, 자체 LLM VARCO 및 생성 AI VARCO Studio 발표
출처 : https://ncsoft.github.io/ncresearch/varco-llm/

VARCO LLM
23년8.16 중형모델 1.3B, 6.4B,13B 오픈, 현재까지 공개된 유사한 크기 한국어 언어 모델보다 최고 성능
- 9월 예정 13B 한국어/영어 동시 학습 모델, 페르소나 감정/의도, 내러티브/게임 퀘스트 생성 지원 모델
- 11월 예정 52B
- 24년3월 예정 100B 멀티모달 텍스트/그림 이해 응답 초거대 모델
NC자체 인프라 활용 서비스 및 AWS 클라우드 SageMaker 인프라 서비스 제공
VARCO Studio
거대 언어모델을 활용하여 생성 AI를 보다 쉽게 활용하기 위한 도
텍스트 생성 및 관리툴(VARCO Text), 이미지 생성툴(VARCO Art),디지털휴먼 생성 및 편집, 운영툴(VARCO Human) 로 구성
현재 게임 콘텐츠 개발을 위한 도구로 사용중이며, 향후 일반인들도 사용할 수 있도록 할 예정
VARCO 소개 설명 동영상 https://youtu.be/sCv4jql5URY

## StabilityAI 코딩 지원 AI StableCode
```
Stability AI가 코딩을 위한 LLM 생성 AI 제품인 StableCode의 출시를 발표했습니다. 
이 제품은 프로그래머의 일상 업무를 지원하는 동시에 기술을 한 단계 더 발전시킬 준비가 된 신규 개발자에게 훌륭한 학습 도구를 제공하도록 설계되었습니다.
StableCode는 개발자의 코딩을 돕기 위해 세 가지 모델을 사용하여 개발자의 효율성을 높일 수 있는 독특한 방법을 제공합니다. 
기본 모델은 먼저 BigCode의 스택 데이터세트(v1.2)에서 다양한 프로그래밍 언어 세트를 학습한 다음 Python, Go, Java, Javascript, C, 마크다운, C++와 같은 인기 언어로 추가 학습을 거쳤습니다.
총 560억 개의 코드 토큰에 대해 HPC 클러스터에서 모델을 학습시켰습니다.
기본 모델이 확립된 후에는 복잡한 프로그래밍 작업을 해결하는 데 도움이 되도록 특정 사용 사례에 맞게 명령어 모델을 조정했습니다. 
알파카 형식의 약 120,000개의 코드 명령어/응답 쌍을 기본 모델에 학습시켜 이 결과를 얻었습니다.
StableCode는 코딩에 대해 더 많이 배우고 싶은 분들에게 이상적인 빌딩 블록이며, 긴 컨텍스트 창 모델은 사용자가 한 줄 및 여러 줄 자동 완성 제안을 사용할 수 있도록 하는 완벽한 보조 도구입니다.
이 모델은 한 번에 훨씬 더 많은 코드를 처리할 수 있도록 설계되어(컨텍스트 창이 16,000토큰인 이전에 출시된 개방형 모델보다 2~4배 더 많음) 사용자가 동시에 최대 5개의 평균 크기 Python 파일을 검토하거나 편집할 수 있으므로 더 큰 도전에 나서고자 하는 초보자에게 이상적인 학습 도구입니다.
다음은 비슷한 수의 매개변수와 훈련된 토큰 수를 가진 다른 모델과 비교하는 방법입니다. 
널리 사용되는 HumanEval 벤치마크를 사용하여 표준 pass@1 및 pass@10 메트릭을 사용합니다.
Stability AI는 기술의 접근성을 높이는 것을 목표로 하며, StableCode는 이 목표를 향한 중요한 단계입니다. 
모든 배경을 가진 사람들이 곧 AI를 사용하여 일상적인 문제를 해결하고 삶을 개선하는 코드를 만들 수 있게 될 것이며, 저희는 이를 실현하는 데 도움이 되고자 합니다.
StableCode를 통해 향후 10억 명의 소프트웨어 개발자가 코딩을 배우는 동시에 전 세계 모든 사람들이 기술에 더 공정하게 접근할 수 있기를 바랍니다.
```
https://huggingface.co/stabi.../stablecode-instruct-alpha-3b

## Denoising MCMC for Accelerating Diffusion-Based Generative Models
Conference: ICML 2023 (Oral Paper)

Author: Beomsu Kim (KAIST AI), Jong Chul Ye (KAIST AI)
```
한국어 개요: Diffusion model은 높은 퀄리티의 데이터를 생성할 수 있으나 생성 속도가 느리다는 단점이 있다. 본 연구에서는 Markov Chain Monte Carlo와 diffusion model을 합침으로써 생성 데이터의 퀄리티를 보존하며 생성 속도를 비약적으로 높일 수 있음을 보였다. 제안한 방법론을 통해 CIFAR10과 CelebA-HQ-256 데이터 생성에서 SOTA 성능을 달성하였으며, FFHQ-1024와 같은 고화질 이미지 생성 가속도 가능함을 실험적으로 보였다.
Abstract : The sampling process of diffusion models can be interpreted as solving the reverse stochastic differential equation (SDE) or the ordinary differential equation (ODE) of the diffusion process, which often requires up to thousands of discretization steps to generate a single image. This has sparked a great interest in developing efficient integration techniques for reverse-S/ODEs. Here, we propose an orthogonal approach to accelerating score-based sampling: Denoising MCMC (DMCMC). DMCMC first uses MCMC to produce initialization points for reverse-S/ODE in the product space of data and diffusion time. Then, a reverse-S/ODE integrator is used to denoise the initialization points. Since MCMC traverses close to the data manifold, the cost of producing a clean sample for DMCMC is much less than that of producing a clean sample from noise. Denoising Langevin Gibbs, an instance of DMCMC, successfully accelerates all six reverse-S/ODE integrators considered in this work, and achieves state-of-the-art results: in the limited number of score function evaluation (NFE) setting on CIFAR10, we have 3.25 FID with 10 NFE and 2.49 FID with 16 NFE. On CelebA-HQ-256, we have 6.99 FID with 160 NFE, which beats the current best record of Kim et al. (2022) among score-based models, 7.16 FID with 4000 NFE.
```
Slides: https://www.slideshare.net/.../denoising-mcmc-for...
Code: https://github.com/1202kbs/DMCMC
Source: https://arxiv.org/abs/2209.14593

## SDXL 1.0 
Stability AI 팀은 텍스트-이미지 생성 모델의 진화에서 다음 반복인 개방형 모델 SDXL 1.0으로 출시하게 된 것을 자랑스럽게 생각합니다. SDXL 0.9의 제한된 연구 전용 릴리스에 이어 SDXL의 정식 버전은 세계 최고의 개방형 이미지 생성 모델로 개선되었습니다.
- Stability AI의 최고의 이미지 모델
SDXL은 거의 모든 아트 스타일에서 고품질의 이미지를 생성하며 포토리얼리즘을 위한 최고의 오픈 모델입니다. 모델에 특정한 '느낌'을 부여하지 않고도 뚜렷한 이미지를 구현할 수 있어 스타일에 대한 절대적인 자유를 보장합니다. 특히 SDXL 1.0은 기본 1024x1024 해상도에서 이전 버전보다 더 나은 콘트라스트, 조명 및 그림자를 통해 생생하고 정확한 색상을 구현하도록 잘 조정되어 있습니다.
또한 SDXL은 손과 텍스트 또는 공간적으로 배치된 구도(예: 전경의 개를 쫓는 배경의 여성)와 같이 이미지 모델로 렌더링하기 어려운 컨셉을 생성할 수 있습니다.
-도전적인 컨셉과 스타일을 위한 더 나은 아트워크
SDXL은 복잡하고 디테일하며 미학적으로 만족스러운 이미지를 몇 마디만 입력하면 만들 수 있습니다. 사용자는 더 이상 고품질 이미지를 얻기 위해 '걸작'과 같은 한정어를 호출할 필요가 없습니다. 또한 SDXL은 '붉은 광장'(유명한 장소)과 '붉은 사각형'(도형)과 같은 개념 간의 차이점을 이해할 수 있습니다.
- 더 간단한 언어로 지능화
SDXL은 복잡하고 디테일하며 미학적으로 만족스러운 이미지를 몇 마디만 입력하면 만들 수 있습니다. 사용자는 더 이상 고품질 이미지를 얻기 위해 '걸작'과 같은 한정어를 호출할 필요가 없습니다. 또한 SDXL은 '붉은 광장'(유명한 장소)과 '붉은 사각형'(도형)과 같은 개념 간의 차이점을 이해할 수 있습니다.
- 가장 큰 오픈 이미지 모델
SDXL 1.0은 35억 개의 파라미터 기본 모델과 66억 개의 파라미터 리파이너로 구성된 혁신적인 새 아키텍처를 기반으로 구축되어 오픈 액세스 이미지 모델 중 가장 많은 파라미터 수를 보유하고 있습니다. 전체 모델은 잠재적 확산을 위한 전문가 혼합 파이프라인으로 구성됩니다: 첫 번째 단계에서는 기본 모델이 (노이즈가 있는) 잠상을 생성한 다음 최종 노이즈 제거 단계에 특화된 정제 모델을 사용하여 추가로 처리합니다. 기본 모델은 독립형 모듈로도 사용할 수 있습니다. 이 2단계 아키텍처는 속도 저하나 과도한 컴퓨팅 리소스를 요구하지 않으면서도 강력한 이미지 생성을 가능하게 합니다. SDXL 1.0은 8GB VRAM이 탑재된 소비자용 GPU 또는 즉시 사용 가능한 클라우드 인스턴스에서 효과적으로 작동합니다.
- 미세 조정 및 고급 제어
SDXL 1.0을 사용하면 사용자 지정 데이터에 맞게 모델을 미세 조정하는 것이 그 어느 때보다 쉬워집니다. 데이터 랭글링 없이도 사용자 지정 LoRA 또는 체크포인트를 생성할 수 있습니다. Stability AI 팀은 SDXL에 특화된 T2I / ControlNet을 통해 차세대 작업별 구조, 스타일 및 구성 제어를 구축하고 있습니다. 이러한 기능은 현재 베타 프리뷰 버전이지만 미세 조정에 대한 업데이트를 계속 지켜봐 주시기 바랍니다.
SDXL의 이미지 제어 기능은 곧 출시될 예정입니다.

### Get started with SDXL
There are several ways to get started with SDXL 1.0:
-SDXL 1.0 is live on Clipdrop. Follow this link.
-The weights of SDXL 1.0 and the associated source code have been released on the Stability AI GitHub page.
https://github.com/Stability-AI/generative-models
-SDXL 1.0 is also being released for API on the Stability AI Platform.
-SDXL 1.0 is available on AWS Sagemaker and AWS Bedrock.
-The Stable Foundation Discord is open for live testing of SDXL models.
-DreamStudio has SDXL 1.0 available for image generation as well.

https://stability.ai/.../stable-diffusion-sdxl-1...

추가 : 바로 다운로드 받을 수 있는 곳
SDXL 1.0 base
https://huggingface.co/stabi.../stable-diffusion-xl-base-1.0
SDXL 1.0 refiner
https://huggingface.co/.../stable-diffusion-xl-refiner-1.0

## FABRIC: Personalizing Diffusion Models with Iterative Feedback (23.7, ETH Zürich, Switzerland)
디퓨전 이미지 생성에서 훈련없이 사용자의 피드백(좋아요,싫어요)으로 출력 결과를 원하는 방향으로 조정 가능
1. Image Gen. ->
2. Result Images Like/Unlike check ->
3. Re-Gen.(same prompt) ->
4. liked style Imgaes Gen.
- Paper: https://arxiv.org/abs/2307.10159
- Project: https://sd-fabric.github.io
- code: https://github.com/sd-fabric/fabric
(deepl 번역) 이 연구에서는 사용자 경험과 출력 품질을 향상시키기 위해 반복적인 인간 피드백을 확산 기반 텍스트-이미지 모델에 통합하는 방법을 살펴봅니다. 광범위한 확산 모델에 적용할 수 있는 피드백 이미지 세트에 대한 확산 프로세스를 조건화하는 훈련이 필요 없는 접근 방식인 FABRIC(주의 기반 참조 이미지 컨디셔닝을 통한 피드백)을 소개합니다. 이 접근법을 엄격하게 평가하기 위한 포괄적인 평가 방법론을 제안하고, 여러 차례의 반복적인 피드백을 통해 생성 결과가 개선되어 사용자 선호도를 최적화한다는 사실을 입증합니다. 이 연구는 개인화된 콘텐츠 제작 및 커스터마이징에 잠재적으로 적용되어 텍스트-이미지 생성 연구의 발전에 기여할 수 있습니다.

## TokenFlow: Consistent Diffusion Features for Consistent Video Editing (23.7, Weizmann institute of science)
일관된 비디오 편집을 위한 일관된 확산(Diffusion) 기능

(deepl번역) 
```
제너레이티브 AI 혁명은 최근 동영상으로 확장되고 있습니다.
그럼에도 불구하고 현재의 최첨단 비디오 모델은 시각적 품질과 생성된 콘텐츠에 대한 사용자 제어 측면에서 이미지 모델에 비해 여전히 뒤쳐져 있습니다.
이 연구에서는 텍스트 기반 비디오 편집 작업을 위해 텍스트-이미지 확산 모델의 힘을 활용하는 프레임워크를 제시합니다.
구체적으로, 소스 비디오와 타겟 텍스트 프롬프트가 주어지면 입력 비디오의 공간 레이아웃과 모션을 유지하면서 타겟 텍스트에 맞는 고품질 비디오를 생성하는 방법을 제시합니다.
이 방법은 확산 특징 공간에 일관성을 적용함으로써 편집된 비디오의 일관성을 얻을 수 있다는 핵심 관찰에 기반합니다.
모델에서 쉽게 사용할 수 있는 프레임 간 대응을 기반으로 확산 특징을 명시적으로 전파함으로써 이를 달성합니다.
따라서 프레임워크는 별도의 교육이나 미세 조정이 필요하지 않으며, 기성 텍스트-이미지 편집 방법과 함께 사용할 수 있습니다.
다양한 실제 동영상에 대한 최첨단 편집 결과를 시연합니다.
```
- project : https://diffusion-tokenflow.github.io/
- paper: https://arxiv.org/abs/2307.10373
- code : https://github.com/omerbt/TokenFlow (comming soon)

## HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models (23.7,  google)

드림부스보다 25배, 텍스트 반전(Textual Inversion)보다 125배 빠른 약 20초 만에 단 하나의 레퍼런스 이미지를 사용하여 드림부스와 동일한 품질과 다양한 스타일로 얼굴에 개인화를 구현, 일반 드림부스 모델보다 10000배 더 작은 모델

```
(deepl 번역)
개인화는 다양한 상황과 스타일의 개인을 다양한 맥락과 스타일로 합성하면서도 정체성을 충실하게 유지할 수 있는 제너레이티브 AI 분야에서 중요한 측면으로 부상하고 있습니다.
그러나 개인화 프로세스에는 시간과 메모리 요구 사항이라는 본질적인 과제가 있습니다.
각 개인화 모델을 미세 조정하려면 상당한 GPU 시간을 투자해야 하며, 피사체별로 개인화 모델을 저장하려면 스토리지 용량이 많이 필요할 수 있습니다.
이러한 문제를 극복하기 위해 우리는 사람의 단일 이미지에서 개인화된 가중치 세트를 효율적으로 생성할 수 있는 하이퍼네트워크인 하이퍼드림부스(HyperDreamBooth)를 제안합니다.
이러한 가중치를 확산 모델에 구성하고 빠른 미세 조정을 통해 HyperDreamBooth는 다양한 스타일과 의미적 수정에 대한 모델의 중요한 지식을 보존하면서 피사체 디테일이 높은 다양한 컨텍스트와 스타일의 인물 얼굴을 생성할 수 있습니다.
우리의 방식은 드림부스보다 25배, 텍스트 반전(Textual Inversion)보다 125배 빠른 약 20초 만에 단 하나의 레퍼런스 이미지를 사용하여 드림부스와 동일한 품질과 다양한 스타일로 얼굴에 개인화를 구현할 수 있습니다.
또한 이 방법을 사용하면 일반 드림부스 모델보다 10000배 더 작은 모델을 생성할 수 있습니다. 
```

- paper : https://arxiv.org/abs/2307.06949
- project : https://hyperdreambooth.github.io/
  
## Do We Still Need Clinical Language Models?
작은 언어 모델을 특정 도메인(의료 분야)의 텍스트 데이터만으로 바닥부터 학습한 게 가장 성능이 좋다는 연구입니다.
대규모의 일반적인 텍스트 데이터에 사전 학습된 이른바 파운데이션 모델을 특정 도메인의 데이터로 파인튜닝해서 사용하는 방법이 일종의 표준이 된 것에 반하는 연구 결과입니다. 
  - 논문 https://arxiv.org/abs/2302.08091

## Unleashing Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration

```
논문 초록
인간의 지능은 서로 다른 인지 과정 간의 협업과 정보 통합이 개별적인 인지 과정의 고립에 비해 더 나은 결과를 낳는 인지 시너지 개념을 바탕으로 번창합니다. 대규모 언어 모델(LLM)은 일반적인 과제 해결 에이전트로서 유망한 성능을 입증했지만, 집중적인 도메인 지식과 복잡한 추론이 필요한 과제에서는 여전히 어려움을 겪고 있습니다. 이 연구에서는 단일 LLM을 여러 페르소나와 멀티턴 셀프 협업에 참여시켜 인지적 시너지 효과로 전환하는 솔로 성능 프롬프트(SPP)를 제안합니다. 인지적 시너지란 여러 사람과 협업하여 각자의 강점과 지식을 결합하여 복잡한 작업에서 문제 해결 및 전반적인 성과를 향상시키는 지능형 에이전트를 말합니다. SPP는 작업 입력에 따라 다양한 페르소나를 동적으로 식별하고 시뮬레이션함으로써 LLM에서 인지적 시너지의 잠재력을 발휘합니다. 우리는 LLM에서 여러 개의 세분화된 페르소나를 할당하면 단일 또는 고정된 수의 페르소나를 사용하는 것보다 더 나은 문제 해결 능력을 이끌어낸다는 사실을 발견했습니다. 세 가지 도전적인 과제를 통해 SPP를 평가합니다: 지식 집약적 유형과 추론 집약적 유형을 모두 아우르는 퀴즈 창의적 글쓰기, 코드네임 협업, 논리 격자 퍼즐이 그것입니다. 연쇄사고력(Chain-of-Thought)과 같이 단순히 추론 능력만 강화하는 기존 작업과 달리, SPP는 내적 지식 습득 능력을 효과적으로 이끌어내고, 환각을 줄이며, 강력한 추론 능력을 유지하도록 합니다.
```
- 논문 https://arxiv.org/abs/2307.05300
- 깃허브 https://github.com/MikeWangWZHL/Solo-Performance-Prompting

## AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning (23.7, Shanghai AI Laboratory외)
특별한 튜닝 없이 개인화된 텍스트-to-이미지 디퓨전 모델에 애니메이션 적용하기
paper : https://arxiv.org/abs/2307.04725
site : https://animatediff.github.io
소스 : https://github.com/guoyww/animatediff/
(설명, deepl 번역) 텍스트-대-이미지 모델(예: Stable Diffusion)과 이에 대응하는 개인화 기술(예: DreamBooth, LoRA)의 발전으로 누구나 저렴한 비용으로 자신의 상상력을 고품질 이미지로 구현할 수 있게 되었습니다. 이에 따라 생성된 정적 이미지에 모션 다이내믹스를 더하기 위한 이미지 애니메이션 기법에 대한 요구가 커지고 있습니다. 본 보고서에서는 기존의 대부분의 개인화된 텍스트-이미지 모델을 한 번에 애니메이션화할 수 있는 실용적인 프레임워크를 제안하여 모델별 튜닝에 대한 수고를 덜어줍니다.
제안된 프레임워크의 핵심은 고정된 텍스트-이미지 모델에 새로 초기화된 모션 모델링 모듈을 삽입하고 이를 비디오 클립에 학습시켜 합리적인 모션 프리퍼를 추출하는 것입니다. 학습이 완료되면 이 모션 모델링 모듈을 삽입하기만 하면 동일한 기본 T2I에서 파생된 모든 개인화된 버전이 텍스트 기반 모델이 되어 다양하고 개인화된 애니메이션 이미지를 쉽게 생성할 수 있습니다.
우리는 애니메이션 사진과 실제 사진에 걸쳐 몇 가지 대표적인 공개 개인화 텍스트-이미지 모델에 대한 평가를 수행하고, 우리가 제안한 프레임워크가 이러한 모델이 출력물의 영역과 다양성을 유지하면서 시간적으로 부드러운 애니메이션 클립을 생성하는 데 도움이 된다는 것을 입증합니다.

소스에서 확인한 내용
1) 준비 사항
- 인퍼런스에 약 60GB 필요, NVIDIA A100 추천 (개인용 GPU로는 불가....)
- Base T2I 체크포인트 다운 : stable-diffusion-v1-4, v1-5
- Motion 체크포인트 다운 : mm_sd_v15.ckpt , mm_sd_v14.ckpt (이게 이 논문에서 학습된 핵심 체크포인트 인 듯)
- Prepare Personalize T2I 다운(civitai 체크포인트): ToonYou, RealisticVision ...
2) 실행(인퍼런스) 명령
- python -m scripts.animate --config configs/prompts/1-ToonYou.yaml
python -m scripts.animate --config configs/prompts/5-RealisticVision.yaml

## Thought Cloning: Learning to Think while Acting by Imitating Human Thinking (University of British Columbia, June 2023)
Paper: https://arxiv.org/abs/2306.00323
Abstract:
"Language is often considered a key aspect of human thinking, providing us with exceptional abilities to generalize, explore, plan, replan, and adapt to new situations. However, Reinforcement Learning (RL) agents are far from human-level performance in any of these abilities. We hypothesize one reason for such cognitive deficiencies is that they lack the benefits of thinking in language and that we can improve AI agents by training them to think like humans do. We introduce a novel Imitation Learning framework, Thought Cloning, where the idea is to not just clone the behaviors of human demonstrators, but also the thoughts humans have as they perform these behaviors. While we expect Thought Cloning to truly shine at scale on internet-sized datasets of humans thinking out loud while acting (e.g. online videos with transcripts), here we conduct experiments in a domain where the thinking and action data are synthetically generated. Results reveal that Thought Cloning learns much faster than Behavioral Cloning and its performance advantage grows the further out of distribution test tasks are, highlighting its ability to better handle novel situations. Thought Cloning also provides important benefits for AI Safety and Interpretability, and makes it easier to debug and improve AI. Because we can observe the agent's thoughts, we can (1) more easily diagnose why things are going wrong, making it easier to fix the problem, (2) steer the agent by correcting its thinking, or (3) prevent it from doing unsafe things it plans to do. Overall, by training agents how to think as well as behave, Thought Cloning creates safer, more powerful agents."
- GitHub: https://github.com/ShengranHu/Thought-Cloning
- Article: https://bdtechtalks.com/2023/07/03/ai-thought-cloning/

## LEDITS: Real Image Editing with DDPM Inversion and Semantic Guidance (23.7, HuggingFace)
이미지에서 텍스트로 특정 물체를 추가 및 삭제하거나 스타일을 바꾸

첫번째 첨부 이미지는 직접 데모에서 해본 것, 원본 얼굴 이미지 넣고, "glasses" 와 "painting" 추가해 본 후 결과
(수정 추가) 두번째 첨부 이미지는 권투 인물 사진 넣고 "Elon Musk" face 입력 결과.  
세번째 첨부 이미지는 논문에 있는 이미지
paper: https://arxiv.org/abs/2307.00522
project: https://editing-images-project.hf.space/index.html
demo: https://huggingface.co/spaces/editing-images/ledits
(논문 설명 deepl 번역)
최근의 대규모 텍스트 유도 확산(diffusion) 모델은 강력한 이미지 생성 기능을 제공합니다. 현재는 직관적이고 다양한 편집을 제공하기 위해 텍스트만을 사용하여 이러한 이미지를 수정할 수 있도록 하는 데 많은 노력을 기울이고 있습니다. 그러나 원본 이미지의 특정 콘텐츠를 보존해야 하는 편집 기술의 본질적인 특성으로 인해 이러한 생성 모델에서는 편집이 어려운 것으로 나타났습니다. 반대로 텍스트 기반 모델에서는 텍스트 프롬프트를 조금만 수정해도 전혀 다른 결과가 나오는 경우가 많기 때문에 사용자의 의도에 정확하게 부합하는 원샷 생성을 달성하는 것이 매우 어렵습니다. 또한 이러한 최첨단 툴을 사용하여 실제 이미지를 편집하려면 먼저 이미지를 사전 학습된 모델 영역으로 반전시켜야 하므로 지연 시간뿐만 아니라 편집 품질에 영향을 미치는 또 다른 요소가 추가됩니다. 이 탐색 보고서에서는 실제 이미지 편집을 위한 경량 접근 방식인 LEDITS를 제안하며, 편집 친화적인 DDPM 반전 기법과 시맨틱 가이던스를 통합하여 시맨틱 가이던스를 실제 이미지 편집으로 확장하는 동시에 DDPM 반전의 편집 기능도 활용할 수 있도록 합니다. 이 접근 방식은 구도 및 스타일 변경은 물론 미묘하고 광범위한 편집을 다양하게 수행할 수 있으며 아키텍처를 최적화하거나 확장할 필요가 없습니다.

## Towards Healthy AI: Large Language Models Need Therapists Too

이 논문에서는 AI 챗봇의 유해한 행동을 수정하고 인간의 가치에 부합하도록 개선하기 위해 심리치료를 통합하는 SafeguardGPT 프레임워크를 제안하고, 소셜 대화를 시뮬레이션하는 작업 사례를 통해 그 효과를 입증합니다.
PDF: https://arxiv.org/pdf/2304.00416.pdf
## Transformer/Attention Tutorial/Survey in Other Disciplines
* Everything You Need to Know about Transformers: Architectures, Optimization, Applications, and Interpretation, in *AAAI Tutorial* 2023. [\[link\]](https://transformer-tutorial.github.io/aaai2023/)  
* Transformer Architectures for Multimodal Signal Processing and Decision Making, in *ICASSP Tutorial* 2022. [\[link\]](https://transformer-tutorial.github.io/icassp2022/)  
* Efficient transformers: A survey, in *ACM Computing Surveys* 2022. [\[paper\]](https://dl.acm.org/doi/10.1145/3530811) [\[paper\]](https://arxiv.org/abs/2009.06732)
* A survey on visual transformer, in *IEEE TPAMI* 2022. [\[paper\]](https://arxiv.org/abs/2012.12556)
* A General Survey on Attention Mechanisms in Deep Learning, in *IEEE TKDE* 2022. [\[paper\]](https://personal.eur.nl/frasincar/papers/TKDE2022/tkde2022.pdf)
* Attention, please! A survey of neural attention models in deep learning, in *Artificial Intelligence Review* 2022. [\[paper\]](https://link.springer.com/article/10.1007/s10462-022-10148-x)
* Attention mechanisms in computer vision: A survey, in *Computational Visual Media* 2022. [\[paper\]](https://link.springer.com/article/10.1007/s41095-022-0271-y)
* Survey: Transformer based video-language pre-training, in _AI Open_ 2022. [\[paper\]](https://www.sciencedirect.com/science/article/pii/S2666651022000018)
* Transformers in vision: A survey, in *ACM Computing Surveys* 2021. [\[paper\]](https://arxiv.org/abs/2101.01169)
* Pre-trained models: Past, present and future, in *AI Open* 2021. [\[paper\]](https://www.sciencedirect.com/science/article/pii/S2666651021000231)
* An attentive survey of attention models, in *ACM TIST* 2021. [\[paper\]](https://arxiv.org/abs/1904.02874)
* Attention in natural language processing, in *IEEE TNNLS* 2020. [\[paper\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9194070)
* Pre-trained models for natural language processing: A survey, in *Science China Technological Sciences* 2020. [\[paper\]](https://link.springer.com/article/10.1007/s11431-020-1647-3)
* A review on the attention mechanism of deep learning, in *Neurocomputing* 2021. [\[paper\]](https://www.sciencedirect.com/science/article/abs/pii/S092523122100477X)
* A Survey of Transformers, in _arXiv_ 2021. [\[paper\]](https://arxiv.org/abs/2106.04554)
* A Survey of Vision-Language Pre-Trained Models, in _arXiv_ 2022. [\[paper\]](https://arxiv.org/abs/2202.10936)
* Video Transformers: A Survey, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2201.05991)
* Transformer for Graphs: An Overview from Architecture Perspective, in _arXiv_ 2022. [\[paper\]](https://arxiv.org/abs/2202.08455)
* Transformers in Medical Imaging: A Survey, in _arXiv_ 2022. [\[paper\]](https://arxiv.org/abs/2201.09873) 
* A Survey of Controllable Text Generation using Transformer-based Pre-trained Language Models, in _arXiv_ 2022. [\[paper\]](https://arxiv.org/abs/2201.05337) 

## Segment Anything: 
https://arxiv.org/abs/2304.02643

## OpenAi's GPT-2
 - 345m 짜리 모델을 공개
 - 762m, 1.5b는 일부에게 공개
 - links : https://openai.com/blog/better-language-models
 
## Download LLaMA from meta.
To download all model weights, then run this:

Linux:
```sh
curl -o- https://raw.githubusercontent.com/shawwn/llama-dl/56f50b96072f42fb2520b1ad5a1d6ef30351f23c/llama.sh | bash
```
Mac:
```sh
brew install bash
curl -o- https://raw.githubusercontent.com/shawwn/llama-dl/56f50b96072f42fb2520b1ad5a1d6ef30351f23c/llama.sh | $(brew --prefix)/bin/bash
```

