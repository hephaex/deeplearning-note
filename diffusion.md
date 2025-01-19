# Awesome-Diffusion-Models

This repository contains a collection of resources and papers on ***Diffusion Models***.

*If there are any missing valuable resources or papers or any materials related to diffusion model, please do not hesitate to create or pull request to issues. I am happy to reflect them.*

## Introductory Material
**What are Diffusion Models?** \
*Lilian Weng* \
2021. [[Website](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html)] \
11 Jul 2021 

## Papers
## Image Generation
## TRELLIS(Microsoft/Tsinghua Univ.), text to 3D, image to 3D Generation
래디언스 필드, 3D 가우시안, 메시 등 다양한 출력 포맷으로 디코딩할 수 있는 통합 구조화된 LATent(SLAT) 표현을 이용하여 텍스트 또는 이미지 조건으로 고품질의 결과물을 생성하며, 비슷한 규모의 최근 모델을 포함해 기존 방식을 훨씬 능가.
Structured 3D Latents for Scalable and Versatile 3D Generation (2412,  Tsinghua University / USTC / Microsoft Research)
project : https://trellis3d.github.io/
paper : https://arxiv.org/abs/2412.01506
code : https://github.com/Microsoft/TRELLIS
demo : https://huggingface.co/spaces/JeffreyXiang/TRELLIS
 다목적 고품질 3D 에셋 제작을 위한 새로운 3D 생성 방법을 소개합니다. 그 초석은 래디언스 필드, 3D 가우시안, 메시 등 다양한 출력 포맷으로 디코딩할 수 있는 통합 구조화된 LATent(SLAT) 표현입니다. 이는 밀도가 낮은 3D 그리드와 강력한 비전 기반 모델에서 추출한 고밀도 멀티뷰 시각적 특징을 통합하여 디코딩 중 유연성을 유지하면서 구조(지오메트리) 및 텍스처(외관) 정보를 모두 포괄적으로 캡처함으로써 달성할 수 있습니다. SLAT에 맞게 조정된 정류된 흐름 변환기를 3D 생성 모델로 사용하고 500만 개의 다양한 개체로 구성된 대규모 3D 자산 데이터 세트에서 최대 20억 개의 파라미터로 모델을 훈련합니다. 우리의 모델은 텍스트 또는 이미지 조건으로 고품질의 결과물을 생성하며, 비슷한 규모의 최근 모델을 포함해 기존 방식을 훨씬 능가합니다. 이전 모델에서는 제공하지 않았던 유연한 출력 형식 선택과 로컬 3D 편집 기능을 선보입니다. 코드, 모델 및 데이터가 공개될 예정입니다.

## 많은 분들이 기다리시던 저희 #NAVER #AI_Lab 의 새로운  diffusion 기반의 controllable Text-to-Image 모델 DenseDiffusion 논문과 소스코드가 공개되었습니다. 이 연구는 AI Lab 생성모델팀 김윤지님이 주저자로, 이지영님, 김진화 님, 저 그리고 CycleGAN으로 유명한 생성모델분야 글로벌 최고 연구자 중 1명인 CMU의 Jun-Yan Zhu 교수가 함께 코웍한 연구로 오는 10월 파리에서 열리는 #ICCV23 에서 발표합니다.
추가적인 훈련에 대한 오버헤드 없이 attention moderation 만으로 컨트롤이 강화되고 구체적으로 묘사된 의미의 텍스트 입력도 더욱 정확하게 이미지를 편집가능하므로 많은 분들이 활용해보시면 좋을 것 같네요.
덧으로 Jun-Yan은 학회 제출때는 물론이고 Cam-ready version의 논문 퀄리티 향상을 위해 끝까지 정말 꼼꼼하게 코멘트하고 수정하는 모습을 보여주어 왜 최고의 연구자인지 다시한번 깨닫게 해주었네요 ㅎㅎ
논문:   https://arxiv.org/abs/2308.12964
github: https://github.com/naver-ai/DenseDiffusion

## 2022년 8월, Stable Diffusion 발표로 세상을 놀라게 했던 그때로부터 딱 1년 지났네요.
22년8월10일 Stable Diffusion 런치 발표, 
22년8월22일 Stable Diffusion 모델 오픈 배포를 시작했습니다.
이전에 Dall-E 가 있었고 Midjourney 가 막 오픈베타를 시작하긴 했지만 스테이블 디퓨전의 테스트 이미지들이 꽤 놀라움을 주었습니다.
저도 Dall-E 테스트해 보고 페북에 테스트 이미지 올리기도 했었지만 막대한 비용이 들어가는 이런 LLM 이나 생성모델들이 구글등에서도 오픈하지 않는 상황에서 오픈 모델이 나올지는 생각도 못했읍니다.
이로부터 채 일주일이 지나지 않아 오픈의 힘으로 많은 곳에서 이를 활용한 것들이 쏟아지기 시작했고 SNS 와 뉴스로 들끓었습니다.
"Stable Diffusion은 지금까지 나온 것중 가장 중요한 AI Art 모델"
"Stable Diffusion 공개 1주일만에 벌어진 놀라운 일들"
"Stability AI, 1억 1천만 달러 투자 유치"
22년 9월 인공지능 그림 미술전 1위 논란이 터지면서 더욱더 이미지 생성은 화제가 되었습니다. (이 미술전에 사용된 것은 미드저니였습니다)
오픈된 stable diffusion 을 애니메 이미지들로 파인튜닝한 waifu diffusion 등 파인튜닝 모델들이 나오기 시작하면서 더욱 다양화되고 질 좋은 수정된 수 많은 모델들이 사람들로부터 만들어져 나오기 시작했습니다.
이후 6개월 동안 기술적으로도 놀랍도록 발전되어 나갔습니다.
DreamBooth, LoRA, xformers, ControlNet ...
그러나 2023년 1월에는 무단 이미지 사용 혐의로 소송이 제기되기도 하였습니다.
또한, 상용 미드저니의 놀라운 버전업과 화질 향상이 이어지는 반면
stable diffusion 2.0, 2.1 버전이 나왔지만 1.5 보다 크게 향상을 보여주지는 못했고, 사람들 활용도 덜했으며, 기본 모델들보다 오히려 사람들이 파인튜닝한 모델들이 더 많이 활용되었습니다.
2023년 7월에 고화질용 버전 SDXL 발표로 그동안 상용 미드저니 등에 품질이 떨어진다는 평을 듣던 것에서 다시 도약했다는 말을 듣기 시작했습니다.
1년만에 놀랍도록 화질이나 기술적 측면에서 발전해 왔습니다.
2023년에는 Gen-2, Pika Labs, Zeroscope 등 Text-to-Video 과 Text-to-3D 등이 나오면서 아마도 이런 것들을 준비하고 있는 것으로 보이며 추후에 이런 것들이 발표되지 않을까 싶습니다.

**Bilateral Denoising Diffusion Models** \
*Max W. Y. Lam, Jun Wang, Rongjie Huang, Dan Su, Dong Yu* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2108.11514)] [[Project](https://bilateral-denoising-diffusion-model.github.io)] \
26 Aug 2021

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021

**SDEdit: Image Synthesis and Editing with Stochastic Differential Equations** \
*Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2108.01073)] [[Project](https://sde-image-editing.github.io/)] [[Github](https://github.com/ermongroup/SDEdit)] \
2 Aug 2021 

**Structured Denoising Diffusion Models in Discrete State-Spaces** \
*Jacob Austin<sup>1</sup>, Daniel D. Johnson<sup>1</sup>, Jonathan Ho, Daniel Tarlow, Rianne van den Berg* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.03006)] \
7 Jul 2021 

**Variational Diffusion Models** \
*Diederik P. Kingma<sup>1</sup>, Tim Salimans<sup>1</sup>, Ben Poole, Jonathan Ho* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.00630)] [[Github](https://github.com/revsic/jax-variational-diffwave)] \
1 Jul 2021 

**Non Gaussian Denoising Diffusion Models** \
*Eliya Nachmani<sup>1</sup>, Robin San Roman<sup>1</sup>, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.07582)] [[Project](https://enk100.github.io/Non-Gaussian-Denoising-Diffusion-Models/)] \
14 Jun 2021 

**D2C: Diffusion-Denoising Models for Few-shot Conditional Generation** \
*Abhishek Sinha<sup>1</sup>, Jiaming Song<sup>1</sup>, Chenlin Meng, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.06819)] [[Project](https://d2c-model.github.io/)] [[Github](https://github.com/d2c-model/d2c-model.github.io)] \
12 Jun 2021 

**Learning to Efficiently Sample from Diffusion Probabilistic Models** \
*Daniel Watson, Jonathan Ho, Mohammad Norouzi, William Chan* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.03802)] \
07 Jun 2021 

**A Variational Perspective on Diffusion-Based Generative Models and Score Matching** \
*Chin-Wei Huang, Jae Hyun Lim, Aaron Courville* \
ICML Workshop 2021. [[Paper](https://arxiv.org/abs/2106.02808)] [[Github](https://github.com/CW-Huang/sdeflow-light)] \
5 Jun 2021 

**On Fast Sampling of Diffusion Probabilistic Models** \
*Zhifeng Kong, Wei Ping* \
ICML Workshop 2021. [[Paper](https://arxiv.org/abs/2106.00132)] [[Github](https://github.com/FengNiMa/FastDPM_pytorch)] \
31 May 2021 

**Cascaded Diffusion Models for High Fidelity Image Generation** \
*Jonathan Ho<sup>1</sup>, Chitwan Saharia<sup>1</sup>, William Chan, David J. Fleet, Mohammad Norouzi, Tim Salimans* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.15282)] [[Project](https://cascaded-diffusion.github.io/)] \
30 May 2021 

**Diffusion Models Beat GANs on Image Synthesis** \
*Prafulla Dhariwal<sup>1</sup>, Alex Nichol<sup>1</sup>* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.05233)] [[Github](https://github.com/openai/guided-diffusion)] \
11 May 2021 

**Image Super-Resolution via Iterative Refinement** \
*Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.07636)] [[Project](https://iterative-refinement.github.io/)] [[Github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] \
15 Apr 2021 

**Noise Estimation for Generative Diffusion Models** \
*Robin San-Roman<sup>1</sup>, Eliya Nachmani<sup>1</sup>, Lior Wolf* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.02600)] \
6 Apr 2021 

**Diffusion Probabilistic Models for 3D Point Cloud Generation** \
*Shitong Luo, Wei Hu* \
CVPR 2021. [[Paper](https://arxiv.org/abs/2103.01458)] [[Github](https://github.com/luost26/diffusion-point-cloud)] \
2 Mar 2021 

**Improved Denoising Diffusion Probabilistic Models** \
*Alex Nichol<sup>1</sup>, Prafulla Dhariwal<sup>1</sup>* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2102.09672)] [[Github](https://github.com/openai/improved-diffusion)] \
18 Feb 2021 

**Maximum Likelihood Training of Score-Based Diffusion Models** \
*Yang Song<sup>1</sup>, Conor Durkan<sup>1</sup>, Iain Murray, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2101.09258)] \
22 Jan 2021 

**Learning Energy-Based Models by Diffusion Recovery Likelihood** \
*Ruiqi Gao, Yang Song, Ben Poole, Ying Nian Wu, Diederik P. Kingma* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2012.08125)] [[Github](https://github.com/ruiqigao/recovery_likelihood)] \
15 Dec 2020 

**Score-Based Generative Modeling through Stochastic Differential Equations** \
*Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole* \
ICLR 2021 (Oral). [[Paper](https://arxiv.org/abs/2011.13456)] [[Github](https://github.com/yang-song/score_sde)] \
26 Nov 2020 

**Denoising Diffusion Implicit Models**  \
*Jiaming Song, Chenlin Meng, Stefano Ermon* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2010.02502)] [[Github](https://github.com/ermongroup/ddim)] \
6 Oct 2020

**Denoising Diffusion Probabilistic Models** \
*Jonathan Ho, Ajay Jain, Pieter Abbeel* \
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2006.11239)] [[Github](https://github.com/hojonathanho/diffusion)] [[Github](https://github.com/pesser/pytorch_diffusion)] \
19 Jun 2020 

**Improved Techniques for Training Score-Based Generative Models** \
*Yang Song, Stefano Ermon* \
NeurIPS 2020. [[Paper](https://arxiv.org/abs/2006.09011)] [[Github](https://github.com/ermongroup/ncsnv2)] \
16 Jun 2020 

**Generative Modeling by Estimating Gradients of the Data Distribution** \
*Yang Song, Stefano Ermon* \
NeurIPS 2019. [[Paper](https://arxiv.org/abs/1907.05600)] [[Project](https://yang-song.github.io/blog/2021/score/)] [[Github](https://github.com/ermongroup/ncsn)] \
12 Jul 2019 

**Neural Stochastic Differential Equations: Deep Latent Gaussian Models in the Diffusion Limit** \
*Belinda Tzen, Maxim Raginsky* \
arXiv 2019. [[Paper](https://arxiv.org/abs/1905.09883)] \
23 May 2019 

**Deep Unsupervised Learning using Nonequilibrium Thermodynamics** \
*Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli* \
ICML 2015. [[Paper](https://arxiv.org/abs/1503.03585)] [[Github](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)] \
2 Mar 2015

**A Connection Between Score Matching and Denoising Autoencoders** \
*Pascal Vincent* \
Neural Computation 2011. [[Paper](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)] \
7 Jul 2011

## Super Resolution

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021 

**Cascaded Diffusion Models for High Fidelity Image Generation**  \
*Jonathan Ho<sup>1</sup>, Chitwan Saharia<sup>1</sup>, William Chan, David J. Fleet, Mohammad Norouzi, Tim Salimans* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.15282)] [[Project](https://cascaded-diffusion.github.io/)] \
30 May 2021

**Image Super-Resolution via Iterative Refinement**  \
*Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, Mohammad Norouzi* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.07636)] [[Project](https://iterative-refinement.github.io/)] [[Github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)] \
15 Apr 2021


## Image-to-Image Translation

**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021

**UNIT-DDPM: UNpaired Image Translation with Denoising Diffusion Probabilistic Models**  \
*Hiroshi Sasaki, Chris G. Willcocks, Toby P. Breckon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2104.05358)] \
12 Apr 2021


## Image Inpainting
**ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models** \
*Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, Sungroh Yoon* \
ICCV 2021 (Oral). [[Paper](https://arxiv.org/abs/2108.02938)] [[Github](https://github.com/jychoi118/ilvr_adm)] \
6 Aug 2021

**SDEdit: Image Synthesis and Editing with Stochastic Differential Equations**  \
*Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2108.01073)] [[Project](https://sde-image-editing.github.io/)] [[Github](https://github.com/ermongroup/SDEdit)] \
2 Aug 2021


## Audio Generation

**Variational Diffusion Models** \
*Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.00630)] [[Github](https://github.com/revsic/jax-variational-diffwave)] \
1 Jul 2021 

**PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Driven Adaptive Prior** \
*Sang-gil Lee, Heeseung Kim, Chaehun Shin, Xu Tan, Chang Liu, Qi Meng, Tao Qin, Wei Chen, Sungroh Yoon, Tie-Yan Liu* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2106.06406)] [[Project](https://speechresearch.github.io/priorgrad/)] \
11 Jun 2021 

**Symbolic Music Generation with Diffusion Models** \
*Gautam Mittal, Jesse Engel, Curtis Hawthorne, Ian Simon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2103.16091)] [[Code](https://github.com/magenta/symbolic-music-diffusion)]
30 Mar 2021 

**DiffWave with Continuous-time Variational Diffusion Models** \
*Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, Bryan Catanzaro* \
ICLR 2021 [[Paper](https://arxiv.org/abs/2009.09761)] [[Project](https://diffwave-demo.github.io/)] [[Github](https://github.com/lmnt-com/diffwave)]
21 Sep 2020

**DiffWave: A Versatile Diffusion Model for Audio Synthesis** \
*Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli* \
ICML 2021 (Oral) [[Paper](https://arxiv.org/abs/2009.09761)] [[Github](https://github.com/lmnt-com/diffwave)] [[Github2](https://github.com/revsic/jax-variational-diffwave)] \
21 Sep 2020 

**WaveGrad: Estimating Gradients for Waveform Generation** \
*Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, William Chan*\
ICLR 2021. [[Paper](https://arxiv.org/abs/2009.00713)] [[Project](https://wavegrad.github.io/)] [[Github](https://github.com/ivanvovk/WaveGrad)] \
2 Sep 2020 


## Audio Enhancement

**A Study on Speech Enhancement Based on Diffusion Probabilistic Model** \
*Yen-Ju Lu<sup>1</sup>, Yu Tsao<sup>1</sup>, Shinji Watanabe* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.11876)] \
25 Jul 2021

**Restoring degraded speech via a modified diffusion model** \
*Jianwei Zhang, Suren Jayasuriya, Visar Berisha* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.11347)] \
22 Apr 2021

**NU-Wave: A Diffusion Probabilistic Model for Neural Audio Upsampling**  \
*Junhyeok Lee, Seungu Han* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.02321)] [[Project](https://mindslab-ai.github.io/nuwave/)] [[Github](https://github.com/mindslab-ai/nuwave)] \
6 Apr 2021


## Audio Conversion

**DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion**  \
*Songxiang Liu<sup>1</sup>, Yuewen Cao<sup>1</sup>, Dan Su, Helen Meng* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2105.13871)] [[Github](https://github.com/liusongxiang/diffsvc)] \
28 May 2021


## Text-to-Speech

**Diff-TTS: A Denoising Diffusion Model for Text-to-Speech**  \
*Myeonghun Jeong, Hyeongju Kim, Sung Jun Cheon, Byoung Jin Choi, Nam Soo Kim* \
Interspeech 2021. [[Paper](https://arxiv.org/abs/2104.01409)] \
3 Apr 2021


## Time-series Forecasting

**Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting** \
*Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf* \
ICLR 2021. [[Paper](https://arxiv.org/abs/2101.12072)] \
2 Feb 2021 


## Data Imputation

**CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation** \
*Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon* \
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.03502)] \
7 Jul 2021 


## Handwriting Synthesis

**Diffusion models for Handwriting Generation** \
*Troy Luhman<sup>1</sup>, Eric Luhman<sup>1</sup>* \
arXiv 2020. [[Paper](https://arxiv.org/abs/2011.06704)] [[Github](https://github.com/tcl9876/Diffusion-Handwriting-Generation)] \
13 Nov 2020 
