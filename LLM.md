## MS, 트랜스포머 성능 개선하는 새로운 LLM 아키텍처, Diff Transformer 공개
'트랜스포머' 기반 대형언어모델(LLM)의 긴 컨텍스트 정보 검색 기능을 개선하는 새로운 아키텍처가 나왔다. 
벤처비트는 16일(현지시간) 마이크로소프트(MS)와 칭화대학교 연구진이 관련 컨텍스트에 대한 어텐션(attention)을 증폭하고 노이즈를 걸러내 성능을 개선하는 새로운 아키텍처 ‘차등 트랜스포머(Diff Transformer)’에 관한 논문을 아카이브에 게재했다고 보도했다.
트랜스포머 아키텍처는 대부분 LLM의 기반이다. 어텐션 메커니즘을 사용해 입력 텍스트 내 토큰이 출력 생성에 미치는 중요도를 평가하는 방식이다. 어텐션 메커니즘은 벡터 값을 확률 분포로 정규화하는 '소프트맥스(softmax)' 함수를 사용해 입력 시퀀스의 토큰에 어텐션 점수를 할당한다.

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

