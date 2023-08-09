## Denoising MCMC for Accelerating Diffusion-Based Generative Models
Conference: ICML 2023 (Oral Paper)
Author: Beomsu Kim (KAIST AI), Jong Chul Ye (KAIST AI)
한국어 개요: Diffusion model은 높은 퀄리티의 데이터를 생성할 수 있으나 생성 속도가 느리다는 단점이 있다. 본 연구에서는 Markov Chain Monte Carlo와 diffusion model을 합침으로써 생성 데이터의 퀄리티를 보존하며 생성 속도를 비약적으로 높일 수 있음을 보였다. 제안한 방법론을 통해 CIFAR10과 CelebA-HQ-256 데이터 생성에서 SOTA 성능을 달성하였으며, FFHQ-1024와 같은 고화질 이미지 생성 가속도 가능함을 실험적으로 보였다.
Abstract : The sampling process of diffusion models can be interpreted as solving the reverse stochastic differential equation (SDE) or the ordinary differential equation (ODE) of the diffusion process, which often requires up to thousands of discretization steps to generate a single image. This has sparked a great interest in developing efficient integration techniques for reverse-S/ODEs. Here, we propose an orthogonal approach to accelerating score-based sampling: Denoising MCMC (DMCMC). DMCMC first uses MCMC to produce initialization points for reverse-S/ODE in the product space of data and diffusion time. Then, a reverse-S/ODE integrator is used to denoise the initialization points. Since MCMC traverses close to the data manifold, the cost of producing a clean sample for DMCMC is much less than that of producing a clean sample from noise. Denoising Langevin Gibbs, an instance of DMCMC, successfully accelerates all six reverse-S/ODE integrators considered in this work, and achieves state-of-the-art results: in the limited number of score function evaluation (NFE) setting on CIFAR10, we have 3.25 FID with 10 NFE and 2.49 FID with 16 NFE. On CelebA-HQ-256, we have 6.99 FID with 160 NFE, which beats the current best record of Kim et al. (2022) among score-based models, 7.16 FID with 4000 NFE.
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

