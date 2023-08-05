# Delving deep into Generative Adversarial Networks (GANs) 	

---
## DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing
DragGAN 에 이어 이를 Diffusion 모델에도 적용한 DragDiffusion 이 나왔습니다. DragGAN과 마찬가지로 이미지의 한 점을 움직이는 방식으로 생성 이미지의 편집이 가능합니다.
- 논문 https://arxiv.org/abs//2306.14435

```초록
정밀하고 제어 가능한 이미지 편집은 상당한 주목을 받고 있는 까다로운 작업입니다. 최근 DragGAN은 인터랙티브한 포인트 기반 이미지 편집 프레임워크를 구현하고 픽셀 수준의 정밀도로 인상적인 편집 결과를 얻을 수 있습니다. 그러나 이 방법은 생성적 적대 신경망(GAN)을 기반으로 하기 때문에 미리 학습된 GAN 모델의 용량에 따라 일반성이 상한선이 정해져 있습니다. 이 연구에서는 이러한 편집 프레임워크를 확산 모델로 확장하여 DragDiffusion을 제안합니다. 사전 학습된 대규모 확산 모델을 활용함으로써 실제 시나리오에서 인터랙티브 포인트 기반 편집의 적용 가능성을 크게 향상시킵니다. 기존의 대부분의 확산 기반 이미지 편집 방법은 텍스트 임베딩에서 작동하는 반면, DragDiffusion은 확산 잠재력을 최적화하여 정밀한 공간 제어를 달성합니다. 확산 모델은 반복적인 방식으로 이미지를 생성하지만, 한 단계에서 확산 잠재력을 최적화하는 것만으로도 일관된 결과를 생성할 수 있으며, 이를 통해 고품질 편집을 효율적으로 완료할 수 있음을 경험적으로 보여줍니다. 여러 오브젝트, 다양한 오브젝트 카테고리, 다양한 스타일 등 까다로운 다양한 사례에 대한 광범위한 실험을 통해 DragDiffusion의 다목적성과 범용성을 입증했습니다.
```
- project : https://vcai.mpi-inf.mpg.de/projects/DragGAN/
- github : https://github.com/XingangPan/DragGAN

## MusicGen: Simple and Controllable Music Generation (Meta, 2023.6)
- text-to-music, by Meta(facebook) 
 - paper: https://arxiv.org/abs/2306.05284
 - github : https://github.com/facebookresearch/audiocraft

```(DeepL 번역)
조건부 음악 생성이라는 과제를 해결합니다. 압축된 이산 음악 표현, 즉 토큰의 여러 스트림에서 작동하는 단일 언어 모델(LM)인 MusicGen을 소개합니다. 이전 작업과 달리 MusicGen은 효율적인 토큰 인터리빙 패턴과 함께 단일 단계 트랜스포머 LM으로 구성되므로 계층적 또는 업샘플링과 같은 여러 모델을 캐스케이딩할 필요가 없습니다. 이러한 접근 방식에 따라, 뮤직젠이 텍스트 설명이나 멜로디 특징에 따라 고품질 샘플을 생성하는 동시에 생성된 출력을 더 잘 제어할 수 있는 방법을 시연합니다. 자동 및 인간 연구를 모두 고려한 광범위한 경험적 평가를 수행하여 제안된 접근 방식이 표준 텍스트-음악 벤치마크에서 평가된 기준선보다 우수하다는 것을 보여줍니다. 절제 연구를 통해 뮤직젠을 구성하는 각 구성 요소의 중요성을 조명합니다. 음악 샘플은 보충 자료에서 확인할 수 있습니다.
```

## A curated, quasi-exhaustive list of state-of-the-art publications and resources about Generative Adversarial Networks (GANs) and their applications.
### Background
 Generative models are models that can learn to create data that is similar to data that we give them. 
 One of the most promising approaches of those models are Generative Adversarial Networks (GANs), a branch of unsupervised machine learning implemented by a system of two neural networks competing against each other in a zero-sum game framework. 
 They were first introduced by Ian Goodfellow et al. in 2014. 
 This repository aims at presenting an elaborate list of the state-of-the-art works on the field of Generative Adversarial Networks since their introduction in 2014.
---
##  InstructPix2Pix by University of California, Berkeley Learning to Follow Image Editing Instructions.
Code https://huggingface.co/timbrooks/instruct-pix2pix#example
Project https://www.timothybrooks.com/instruct-pix2pix/
For more: https://www.linkedin.com/in/ibrahim-sobh-phd-8681757/

### :link: Contents
* [ Desnapify](Desnapify is a deep convolutional generative adversarial network (DCGAN) trained to remove Snapchat filters from selfie images.GitHub by Inderpreet Singh: https://github.com/ipsingh06/ml-desnapify)
#artificialintelligence #deeplearning #generativeadversarialnetworks #machinelearning #technologydeep convolutional generative adversarial network (DCGAN) trained to remove Snapchat filters from selfie images.

GitHub by Inderpreet Singh: https://github.com/ipsingh06/ml-desnapify
* UCSD and NVIDIA AI Researchers Propose ‘CoordGAN’: a Novel Disentangled GAN Mode That Produces Dense Correspondence Maps Represented by a Novel Coordinate Space
Paper: https://arxiv.org/pdf/2203.16521.pdf
Project: https://jitengmu.github.io/CoordGAN/

#artificialintelligence #deeplearning #generativeadversarialnetworks #machinelearning #technology
* [Opening Publication](#pushpin-opening-publication)
* [Latest paper from Ian Goodfellow](#fire-latest-paper-from-ian-goodfellow)
* [Papers](#clipboard-papers-descending-order-based-on-google-scholar-citations)
* [Theory](#notebook_with_decorative_cover-theory)
* [Presentations](#nut_and_bolt-presentations)
* [Courses](#books-courses--tutorials--blogs-webpages-unless-other-is-stated)
* [Code / Resources / Models](#package-resources--models-descending-order-based-on-github-stars)
* [Frameworks & Libraries](#electric_plug-frameworks--libraries-descending-order-based-on-github-stars)
---
### In The Latest AI Research, CMU And Adobe Researchers Propose An Elegant Emsembling Mechanism For GAN Training That Improves FID by 1.5x to 2x On The Given Dataset
Paper: https://arxiv.org/pdf/2112.09130.pdf
Github: https://github.com/nupurkmr9/vision-aided-gan
Project: https://www.cs.cmu.edu/~vision-aided-gan/

### :pushpin: Opening Publication 	
Generative Adversarial Nets (GANs) (2014) [[pdf]](https://arxiv.org/pdf/1406.2661v1.pdf)  [[presentation]](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf) [[code]](https://github.com/goodfeli/adversarial) [[video]](https://www.youtube.com/watch?v=HN9NRhm9waY)


### :fire: Latest paper from Ian Goodfellow

Self-Attention Generative Adversarial Networks (SAGAN) (2018) [[pdf]](https://arxiv.org/pdf/1805.08318.pdf) [[PyTorch implementation]](https://github.com/heykeetae/Self-Attention-GAN) 

---

### :clipboard: Papers (Descending order based on Google Scholar Citations)

S/N|Paper|Year|Citations
:---:|:---:|:---:|:---:
|1|Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGANs)  [[pdf]](https://arxiv.org/pdf/1511.06434v2.pdf)|2015|1534
|2|Explaining and Harnessing Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1412.6572.pdf)|2014|895
|3|Improved Techniques for Training GANs  [[pdf]](https://arxiv.org/pdf/1606.03498v1.pdf )|2016|748
|4| :chart_with_upwards_trend: Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)  [[pdf]](https://arxiv.org/pdf/1611.07004.pdf)|2016|706
|5| :chart_with_upwards_trend: Wasserstein GAN (WGAN)  [[pdf]](https://arxiv.org/pdf/1701.07875.pdf)|2017|587
|6| :chart_with_upwards_trend: Conditional Generative Adversarial Nets (CGAN)  [[pdf]](https://arxiv.org/pdf/1411.1784v1.pdf)|2014|566
|7|Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (LAPGAN)  [[pdf]](https://arxiv.org/pdf/1506.05751.pdf)|2015|564
|8| :chart_with_upwards_trend: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)  [[pdf]](https://arxiv.org/pdf/1609.04802.pdf)|2016|549
|9|Semi-Supervised Learning with Deep Generative Models  [[pdf]](https://arxiv.org/pdf/1406.5298v2.pdf )|2014|494
|10| :chart_with_upwards_trend: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)  [[pdf]](https://arxiv.org/pdf/1703.10593.pdf)|2017|438
|11|InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1606.03657)|2016|394
|12|Context Encoders: Feature Learning by Inpainting  [[pdf]](https://arxiv.org/pdf/1604.07379)|2016|390
|13|Generative Adversarial Text to Image Synthesis  [[pdf]](https://arxiv.org/pdf/1605.05396)|2016|368
|14| :chart_with_upwards_trend: Improved Training of Wasserstein GANs (WGAN-GP)  [[pdf]](https://arxiv.org/pdf/1704.00028.pdf)|2017|331
|15|Deep multi-scale video prediction beyond mean square error  [[pdf]](https://arxiv.org/pdf/1511.05440.pdf)|2015|301
|16|Adversarial Autoencoders  [[pdf]](https://arxiv.org/pdf/1511.05644.pdf)|2015|277
|17|Energy-based Generative Adversarial Network (EBGAN)  [[pdf]](https://arxiv.org/pdf/1609.03126.pdf)|2016|238
|18|Autoencoding beyond pixels using a learned similarity metric (VAE-GAN)  [[pdf]](https://arxiv.org/pdf/1512.09300.pdf)|2015|233
|19| :chart_with_upwards_trend: Conditional Image Generation with PixelCNN Decoders  [[pdf]](https://arxiv.org/pdf/1606.05328.pdf)|2015|231
|20|Towards Principled Methods for Training Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1701.04862.pdf)|2017|229
|21|Adversarial Feature Learning (BiGAN)  [[pdf]](https://arxiv.org/pdf/1605.09782v6.pdf)|2016|224
|22| :chart_with_upwards_trend: Stacked Generative Adversarial Networks (SGAN)  [[pdf]](https://arxiv.org/pdf/1612.04357.pdf)|2016|215
|23| :chart_with_upwards_trend: StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1710.10916.pdf)|2017|214
|24|Adversarially Learned Inference (ALI)  [[pdf]](https://arxiv.org/pdf/1606.00704.pdf)|2016|211
|25| :chart_with_upwards_trend: Conditional Image Synthesis with Auxiliary Classifier GANs (AC-GAN)  [[pdf]](https://arxiv.org/pdf/1610.09585.pdf)|2016|202
|26| :chart_with_upwards_trend: Learning from Simulated and Unsupervised Images through Adversarial Training (SimGAN) by Apple  [[pdf]](https://arxiv.org/pdf/1612.07828v1.pdf)|2016|192
|27|f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization  [[pdf]](https://arxiv.org/pdf/1606.00709.pdf)|2016|186
|28|Generating Videos with Scene Dynamics (VGAN)  [[pdf]](http://web.mit.edu/vondrick/tinyvideo/paper.pdf)|2016|179
|29|Generative Visual Manipulation on the Natural Image Manifold (iGAN)  [[pdf]](https://arxiv.org/pdf/1609.03552.pdf)|2016|172
|30| :chart_with_upwards_trend: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling (3D-GAN)  [[pdf]](https://arxiv.org/pdf/1610.07584)|2016|171
|31|Generative Moment Matching Networks  [[pdf]](https://arxiv.org/pdf/1502.02761.pdf)|2015|167
|32| :chart_with_upwards_trend: Coupled Generative Adversarial Networks (CoGAN)  [[pdf]](https://arxiv.org/pdf/1606.07536)|2016|162
|33| :chart_with_upwards_trend: BEGAN: Boundary Equilibrium Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1703.10717.pdf)|2017|162
|34|Practical Black-Box Attacks against Deep Learning Systems using Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1602.02697.pdf)|2016|161
|35|Generating Images with Perceptual Similarity Metrics based on Deep Networks   [[pdf]](https://arxiv.org/pdf/1602.02644)|2016|151
|36| :chart_with_upwards_trend: Improving Variational Inference with Inverse Autoregressive Flow  [[pdf]](https://arxiv.org/pdf/1606.04934)|2016|150
|37|Unsupervised Learning for Physical Interaction through Video Prediction   [[pdf]](https://arxiv.org/pdf/1605.07157)|2016|146
|38|Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks (CatGAN)  [[pdf]](https://arxiv.org/pdf/1511.06390.pdf)|2015|141
|39|Learning to Discover Cross-Domain Relations with Generative Adversarial Networks (DiscoGAN) [[pdf]](https://arxiv.org/pdf/1703.05192.pdf)|2017|135
|40|Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks (MGAN)  [[pdf]](https://arxiv.org/pdf/1604.04382.pdf)|2016|130
|41|SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient    [[pdf]](https://arxiv.org/pdf/1609.05473.pdf)|2016|125
|42|Generative Adversarial Imitation Learning  [[pdf]](https://arxiv.org/pdf/1606.03476)|2016|123
|43| :chart_with_upwards_trend: Adversarial Discriminative Domain Adaptation  [[pdf]](https://arxiv.org/pdf/1702.05464)|2017|123
|44|Generative Image Modeling using Style and Structure Adversarial Networks (S^2GAN)  [[pdf]](https://arxiv.org/pdf/1603.05631.pdf)|2016|121
|45|Unsupervised Cross-Domain Image Generation (DTN)  [[pdf]](https://arxiv.org/pdf/1611.02200.pdf)|2016|116
|46|Synthesizing the preferred inputs for neurons in neural networks via deep generator networks   [[pdf]](https://arxiv.org/pdf/1605.09304)|2016|99
|47|Least Squares Generative Adversarial Networks (LSGAN)  [[pdf]](https://arxiv.org/pdf/1611.04076.pdf)|2016|98
|48|Semantic Image Inpainting with Perceptual and Contextual Losses   [[pdf]](https://arxiv.org/pdf/1607.07539.pdf)|2016|98
|49|Conditional generative adversarial nets for convolutional face generation [[pdf]](https://pdfs.semanticscholar.org/42f6/f5454dda99d8989f9814989efd50fe807ee8.pdf)|2014|95
|50|StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1612.03242.pdf)|2016|90
|51|Mode Regularized Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1612.02136)|2016|89
|52|Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro [[pdf]](https://arxiv.org/pdf/1701.07717)|2017|87
|53|Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space (PPGN)  [[pdf]](https://arxiv.org/pdf/1612.00005.pdf)|2016|85
|54|DualGAN: Unsupervised Dual Learning for Image-to-Image Translation    [[pdf]](https://arxiv.org/pdf/1704.02510.pdf)|2017|85
|55|Training generative neural networks via Maximum Mean Discrepancy optimization  [[pdf]](https://arxiv.org/pdf/1505.03906.pdf)|2015|82
|56|Generating images with recurrent adversarial networks  [[pdf]](https://arxiv.org/pdf/1602.05110.pdf)|2016|81
|57|Semantic Segmentation using Adversarial Networks   [[pdf]](https://arxiv.org/pdf/1611.08408.pdf)|2016|81
|58|Learning What and Where to Draw (GAWWN)  [[pdf]](https://arxiv.org/pdf/1610.02454v1.pdf)|2016|77
|59|Amortised MAP Inference for Image Super-resolution (AffGAN)  [[pdf]](https://arxiv.org/pdf/1610.04490.pdf)|2016|75
|60|Generalization and Equilibrium in Generative Adversarial Nets (GANs)  [[pdf]](https://arxiv.org/pdf/1703.00573)|2017|74
|61|VIME: Variational Information Maximizing Exploration  [[pdf]](https://arxiv.org/pdf/1605.09674)|2016|70
|62|Disentangled Representation Learning GAN for Pose-Invariant Face Recognition   [[pdf]](http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf)|2017|70
|63|Neural Photo Editing with Introspective Adversarial Networks (IAN)  [[pdf]](https://arxiv.org/pdf/1609.07093.pdf)|2016|63
|64|Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks  [[pdf]](https://arxiv.org/pdf/1704.01155)|2017|63
|65|Learning in Implicit Generative Models  [[pdf]](https://arxiv.org/pdf/1610.03483.pdf)|2016|62
|66|Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis (TP-GAN)  [[pdf]](https://arxiv.org/pdf/1704.04086.pdf)|2017|60
|67|On the Quantitative Analysis of Decoder-Based Generative Models  [[pdf]](https://arxiv.org/pdf/1611.04273.pdf)|2016|59
|68|Invertible Conditional GANs for image editing (IcGAN)  [[pdf]](https://arxiv.org/pdf/1611.06355.pdf)|2016|57
|69|Unrolled Generative Adversarial Networks (Unrolled GAN)  [[pdf]](https://arxiv.org/pdf/1611.02163.pdf)|2016|56
|70|Attend, infer, repeat: Fast scene understanding with generative models  [[pdf]](https://arxiv.org/pdf/1603.08575.pdf)|2016|55
|71|Pixel-Level Domain Transfer   [[pdf]](https://arxiv.org/pdf/1603.07442)|2016|54
|72|SEGAN: Speech Enhancement Generative Adversarial Network    [[pdf]](https://arxiv.org/pdf/1703.09452.pdf)|2017|45
|73|MMD GAN: Towards Deeper Understanding of Moment Matching Network [[pdf]](https://arxiv.org/pdf/1705.08584.pdf)|2017|42
|74|Learning a Driving Simulator [[pdf]](https://arxiv.org/pdf/1608.01230)|2016|41
|75|Image De-raining Using a Conditional Generative Adversarial Network (ID-CGAN)  [[pdf]](https://arxiv.org/pdf/1701.05957)|2017|40
|76|Face Aging With Conditional Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.01983.pdf)|2017|39
|77| :chart_with_upwards_trend: Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (AnoGAN)  [[pdf]](https://arxiv.org/pdf/1703.05921.pdf)|2017|39
|78|Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities (LS-GAN)  [[pdf]](https://arxiv.org/pdf/1701.06264.pdf)|2017|38
|79|Adversarial Attacks on Neural Network Policies [[pdf]](https://arxiv.org/pdf/1702.02284.pdf)|2017|37
|80|AdaGAN: Boosting Generative Models  [[pdf]](https://arxiv.org/pdf/1701.02386.pdf)|2017|37
|81|Triple Generative Adversarial Nets (Triple-GAN)  [[pdf]](https://arxiv.org/pdf/1703.02291.pdf)|2017|37
|82|Semantic Image Inpainting with Deep Generative Models [[pdf]](https://arxiv.org/pdf/1607.07539.pdf)|2017|37
|83|A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection   [[pdf]](https://arxiv.org/pdf/1704.03414.pdf)|2017|36
|84|Adversarial Examples for Semantic Segmentation and Object Detection  [[pdf]](https://arxiv.org/pdf/1703.08603.pdf)|2017|35
|85|Generative face completion   [[pdf]](https://arxiv.org/pdf/1704.05838)|2016|34
|86|Age Progression / Regression by Conditional Adversarial Autoencoder [[pdf]](https://arxiv.org/pdf/1702.08423)|2017|32
|87|The Space of Transferable Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1704.03453)|2017|32
|88|Learning to Protect Communications with Adversarial Neural Cryptography [[pdf]](https://arxiv.org/pdf/1610.06918)|2016|31
|89| Perceptual Generative Adversarial Networks for Small Object Detection [[pdf]](https://arxiv.org/abs/1706.05274)|2017|30
|90|Temporal Generative Adversarial Nets (TGAN)  [[pdf]](https://arxiv.org/pdf/1611.06624.pdf)|2016|29
|91|Towards Large-Pose Face Frontalization in the Wild   [[pdf]](https://arxiv.org/pdf/1704.06244)|2016|29
|92|Boundary-Seeking Generative Adversarial Networks (BS-GAN)  [[pdf]](https://arxiv.org/pdf/1702.08431.pdf)|2017|28
|93|McGan: Mean and Covariance Feature Matching GAN    [[pdf]](https://arxiv.org/pdf/1702.08398.pdf)|2017|28
|94|The Cramer Distance as a Solution to Biased Wasserstein Gradients [[pdf]](https://arxiv.org/pdf/1705.10743.pdf)|2017|28
|95|SalGAN: Visual Saliency Prediction with Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1701.01081.pdf)|2016|28
|96|MoCoGAN: Decomposing Motion and Content for Video Generation [[pdf]](https://arxiv.org/pdf/1707.04993.pdf)|2017|27
|97|Imitating Driver Behavior with Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1701.06699)|2017|27
|98|A Connection between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models  [[pdf]](https://arxiv.org/pdf/1611.03852.pdf)|2016|26
|99|Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN (MalGAN)  [[pdf]](https://arxiv.org/pdf/1702.05983.pdf)|2016|26
|100|Maximum-Likelihood Augmented Discrete Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.07983)|2017|26
|101|Neural Face Editing with Intrinsic Image Disentangling  [[pdf]](https://arxiv.org/abs/1704.04131)|2017|26
|102|Pose Guided Person Image Generation [[pdf]](https://arxiv.org/pdf/1705.09368.pdf)|2017|26
|103|Good Semi-supervised Learning that Requires a Bad GAN [[pdf]](https://arxiv.org/abs/1705.09783)|2017|25
|104|Connecting Generative Adversarial Networks and Actor-Critic Methods  [[pdf]](https://arxiv.org/pdf/1610.01945.pdf)|2016|24
|105|LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation   [[pdf]](https://arxiv.org/pdf/1703.01560.pdf)|2017|24
|106|On Convergence and Stability of GANs  [[pdf]](https://arxiv.org/pdf/1705.07215.pdf)|2017|23
|107|C-RNN-GAN: Continuous recurrent neural networks with adversarial training    [[pdf]](https://arxiv.org/pdf/1611.09904.pdf)|2016|22
|108|Towards Diverse and Natural Image Descriptions via a Conditional GAN [[pdf]](https://arxiv.org/pdf/1703.06029.pdf)|2017|22
|109|Full Resolution Image Compression with Recurrent Neural Networks [[pdf]](https://arxiv.org/pdf/1608.05148)|2016|21
|110|Recurrent Topic-Transition GAN for Visual Paragraph Generation (RTT-GAN)  [[pdf]](https://arxiv.org/pdf/1703.07022.pdf)|2017|21
|111|Multi-View Image Generation from a Single-View   [[pdf]](https://arxiv.org/pdf/1704.04886)|2016|21
|112|Adversarial Transformation Networks: Learning to Generate Adversarial Examples  [[pdf]](https://arxiv.org/pdf/1703.09387)|2017|21
|113|Simple Black-Box Adversarial Perturbations for Deep Networks  [[pdf]](https://arxiv.org/pdf/1612.06299)|2016|21
|114|Deep Generative Adversarial Networks for Compressed Sensing (GANCS) Automates MRI  [[pdf]](https://arxiv.org/pdf/1706.00051.pdf)|2017|21
|115|RenderGAN: Generating Realistic Labeled Data    [[pdf]](https://arxiv.org/pdf/1611.01331.pdf)|2016|21
|116|Stabilizing Training of Generative Adversarial Networks through Regularization [[pdf]](https://arxiv.org/abs/1705.09367)|2017|21
|117|CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1705.02355)|2017|20
|118|Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1704.00849.pdf)|2017|19
|119| :chart_with_upwards_trend: Gradient descent GAN optimization is locally stable  [[pdf]](https://arxiv.org/abs/1706.04156)|2017|19
|120|Adversarial Training Methods for Semi-Supervised Text Classification  [[pdf]](https://arxiv.org/abs/1605.07725)|2016|18
|121|Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks (SSL-GAN)  [[pdf]](https://arxiv.org/pdf/1611.06430.pdf)|2016|18
|122|Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation [[pdf]](https://arxiv.org/pdf/1705.00389.pdf)|2017|18
|123|Multi-Agent Diverse Generative Adversarial Networks (MAD-GAN)  [[pdf]](https://arxiv.org/pdf/1704.02906.pdf)|2017|18
|124|ALICE: Towards Understanding Adversarial Learning for Joint Distribution Matching [[pdf]](http://papers.nips.cc/paper/7133-alice-towards-understanding-adversarial-learning-for-joint-distribution-matching.pdf)|2017|18
|125|Cooperative Training of Descriptor and Generator Networks [[pdf]](https://arxiv.org/pdf/1609.09408.pdf)|2016|17
|126|3D Shape Induction from 2D Views of Multiple Objects (PrGAN)  [[pdf]](https://arxiv.org/pdf/1612.05872.pdf)|2016|17
|127|Learning to Generate Images of Outdoor Scenes from Attributes and Semantic Layouts (AL-CGAN)  [[pdf]](https://arxiv.org/pdf/1612.00215.pdf)|2016|17
|128|Conditional CycleGAN for Attribute Guided Face Image Generation [[pdf]](https://arxiv.org/abs/1705.09966)|2017|17
|129|Objective-Reinforced Generative Adversarial Networks (ORGAN) [[pdf]](https://arxiv.org/pdf/1705.10843.pdf)|2017|17
|130|Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models [[pdf]](https://arxiv.org/pdf/1705.10843.pdf)|2017|17
|131|Generative Multi-Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1611.01673.pdf)|2016|16
|132|Learning Representations of Emotional Speech with Deep Convolutional Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1705.02394)|2017|16
|133|Dual Motion GAN for Future-Flow Embedded Video Prediction [[pdf]](https://arxiv.org/pdf/1708.00284.pdf)|2017|16
|134|VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning () [[pdf]](https://arxiv.org/abs/1705.07761)|2017|16
|135|Unsupervised Image-to-Image Translation with Generative Adversarial Networks   [[pdf]](https://arxiv.org/pdf/1701.02676.pdf)|2017|15
|136|Improved generator objectives for GANs  [[pdf]](https://arxiv.org/pdf/1612.02780.pdf)|2017|15
|137|DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data  [[pdf]](https://arxiv.org/abs/1706.02071)|2017|15
|138|CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training   [[pdf]](https://arxiv.org/pdf/1703.10155.pdf)|2017|15
|139|Improved Semi-supervised Learning with GANs using Manifold Invariances [[pdf]](https://arxiv.org/abs/1705.08850)|2017|14
|140|Inverting The Generator Of A Generative Adversarial Network  [[pdf]](https://arxiv.org/pdf/1611.05644)|2016|14
|141|Precise Recovery of Latent Vectors from Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.04782)|2016|14
|142|Comparison of Maximum Likelihood and GAN-based training of Real NVPs [[pdf]](https://arxiv.org/pdf/1705.05263.pdf)|2017|14
|143|Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.01705)|2017|14
|144|Semantically Decomposing the Latent Spaces of Generative Adversarial Networks (SD-GAN) [[pdf]](https://arxiv.org/abs/1705.07904)|2017|14
|145|Adversarial Generation of Natural Language [[pdf]](https://arxiv.org/abs/1705.10929)|2017|14
|146|MAGAN: Margin Adaptation for Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1704.03817.pdf)|2017|13
|147|Reconstruction of three-dimensional porous media using generative adversarial neural networks [[pdf]](https://arxiv.org/pdf/1704.03225)|2017|13
|148|SCAN: Structure Correcting Adversarial Network for Chest X-rays Organ Segmentation  [[pdf]](https://arxiv.org/pdf/1703.08770)|2017|13
|149|Generating Multi-label Discrete Electronic Health Records using Generative Adversarial Networks (MedGAN)  [[pdf]](https://arxiv.org/pdf/1703.06490.pdf)|2017|13
|150|SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation  [[pdf]](https://arxiv.org/abs/1706.01805)|2017|13
|151| :chart_with_upwards_trend: CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms [[pdf]](https://arxiv.org/pdf/1706.07068.pdf)|2017|13
|152|It Takes (Only) Two: Adversarial Generator-Encoder Networks [[pdf]](https://arxiv.org/pdf/1704.02304.pdf)|2017|12
|153|Generative Temporal Models with Memory  [[pdf]](https://arxiv.org/pdf/1702.04649.pdf)|2017|12
|154|Generative Adversarial Residual Pairwise Networks for One Shot Learning  [[pdf]](https://arxiv.org/pdf/1703.08033)|2017|12
|155|Crossing Nets: Combining GANs and VAEs with a Shared Latent Space for Hand Pose Estimation [[pdf]](https://arxiv.org/pdf/1702.03431.pdf)|2017|12
|156|Texture Synthesis with Spatial Generative Adversarial Networks (SGAN)  [[pdf]](https://arxiv.org/pdf/1611.08207.pdf)|2016|12
|157|Semi-Latent GAN: Learning to generate and modify facial images from attributes (SL-GAN)   [[pdf]](https://arxiv.org/pdf/1704.02166.pdf)|2017|12
|158|Adversarial Networks for the Detection of Aggressive Prostate Cancer [[pdf]](https://arxiv.org/pdf/1702.08014)|2017|12
|159|Auto-painter: Cartoon Image Generation from Sketch by Using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1705.01908.pdf)|2017|12
|160|Interactive 3D Modeling with a Generative Adversarial Network [[pdf]](https://arxiv.org/abs/1706.05170)|2017|11
|161|Language Generation with Recurrent Generative Adversarial Networks without Pre-training  [[pdf]](https://arxiv.org/abs/1706.01399)|2017|11
|162|GP-GAN: Towards Realistic High-Resolution Image Blending   [[pdf]](https://arxiv.org/pdf/1703.07195.pdf)|2017|11
|163|Universal Adversarial Perturbations Against Semantic Image Segmentation  [[pdf]](https://arxiv.org/pdf/1704.05712)|2017|10
|164|A General Retraining Framework for Scalable Adversarial Classification [[pdf]](https://arxiv.org/pdf/1604.02606.pdf)|2016|10
|165|Contextual RNN-GANs for Abstract Reasoning Diagram Generation (Context-RNN-GAN)  [[pdf]](https://arxiv.org/pdf/1609.09444.pdf)|2016|10
|166|Adversarial Image Perturbation for Privacy Protection--A Game Theory Perspective  [[pdf]](https://arxiv.org/pdf/1703.09471)|2017|10
|167|Multi-Generator Gernerative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1708.02556.pdf)|2017|10
|168|Optimizing the Latent Space of Generative Networks [[pdf]](https://arxiv.org/pdf/1707.05776.pdf)|2017|10
|169|GANs for Biological Image Synthesis [[pdf]](https://arxiv.org/pdf/1708.04692.pdf)|2017|10
|170|MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation using 1D and 2D Conditions   [[pdf]](https://arxiv.org/pdf/1703.10847.pdf)|2016|10
|171| :chart_with_upwards_trend: Deep Generative Adversarial Compression Artifact Removal  [[pdf]](https://arxiv.org/pdf/1704.02518)|2017|10
|172|Adversarial Deep Structural Networks for Mammographic Mass Segmentation  [[pdf]](https://arxiv.org/abs/1612.05970)|2017|9
|173|Adversarial Training For Sketch Retrieval (SketchGAN)  [[pdf]](https://arxiv.org/pdf/1607.02748.pdf)|2016|9
|174|Perceptual Adversarial Networks for Image-to-Image Transformation  [[pdf]](https://arxiv.org/pdf/1706.09138)|2017|9
|175|PixelGAN Autoencoders [[pdf]](https://arxiv.org/pdf/1706.00531.pdf)|2017|9
|176|Unsupervised Diverse Colorization via Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1702.06674.pdf)|2017|9
|177|TAC-GAN - Text Conditioned Auxiliary Classifier Generative Adversarial Network    [[pdf]](https://arxiv.org/pdf/1703.06412.pdf)|2017|9
|178|Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking (GoGAN)  [[pdf]](https://arxiv.org/pdf/1704.04865.pdf)|2017|8
|179|From source to target and back: symmetric bi-directional adaptive GAN [[pdf]](https://arxiv.org/abs/1705.08824)|2017|8
|180|Robust LSTM-Autoencoders for Face De-Occlusion in the Wild   [[pdf]](https://arxiv.org/pdf/1612.08534.pdf)|2016|8
|181|Ensembles of Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1612.00991.pdf)|2016|8
|182|Bayesian GAN [[pdf]](https://arxiv.org/abs/1705.09558)|2017|8
|183|Outline Colorization through Tandem Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.08834.pdf)|2017|8
|184|Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss  [[pdf]](https://arxiv.org/pdf/1708.00961.pdf)|2017|8
|185|ArtGAN: Artwork Synthesis with Conditional Categorial GANs   [[pdf]](https://arxiv.org/pdf/1702.03410.pdf)|2017|8
|186|Generative Semantic Manipulation with Contrasting GAN [[pdf]](https://arxiv.org/pdf/1708.00315.pdf)|2017|8
|187|Steganographic Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1703.05502)|2017|7
|188|GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data  [[pdf]](https://arxiv.org/pdf/1705.04932.pdf)|2017|7
|189|Generative Adversarial Parallelization  [[pdf]](https://arxiv.org/pdf/1612.04021)|2015|7
|190|CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training  [[pdf]](https://arxiv.org/pdf/1709.02023.pdf)|2017|7
|191|Automatic Liver Segmentation Using an Adversarial Image-to-Image Network [[pdf]](https://arxiv.org/pdf/1707.08037.pdf)|2017|7
|192|Megapixel Size Image Creation using Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1706.00082.pdf)|2017|6
|193|Style Transfer for Sketches with Enhanced Residual U-net and Auxiliary Classifier GAN  [[pdf]](https://arxiv.org/abs/1706.03319)|2017|6
|194|Semantic Image Synthesis via Adversarial Learning [[pdf]](https://arxiv.org/pdf/1707.06873.pdf)|2017|6
|195|WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images    [[pdf]](https://arxiv.org/pdf/1702.07392.pdf)|2017|6
|196|Retinal Vessel Segmentation in Fundoscopic Images with Generative Adversarial Networks  [[pdf]](https://arxiv.org/abs/1706.09318)|2017|6
|197|Representation Learning and Adversarial Generation of 3D Point Clouds [[pdf]](https://arxiv.org/pdf/1707.02392.pdf)|2017|6
|198|3D Object Reconstruction from a Single Depth View with Adversarial Learning [[pdf]](https://arxiv.org/pdf/1708.07969.pdf)|2017|6
|199|Abnormal Event Detection in Videos using Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1708.09644.pdf)|2017|6
|200|Improved Adversarial Systems for 3D Object Generation and Reconstruction  [[pdf]](https://arxiv.org/pdf/1707.09557.pdf)|2017|6
|201|Relaxed Wasserstein with Applications to GANs (RWGAN ) [[pdf]](https://arxiv.org/abs/1705.07164)|2017|5
|202|Generative Adversarial Networks as Variational Training of Energy Based Models (VGAN)  [[pdf]](https://arxiv.org/pdf/1611.01799.pdf)|2017|5
|203|Flow-GAN: Bridging implicit and prescribed learning in generative models [[pdf]](https://arxiv.org/abs/1705.08868)|2017|5
|204|Weakly Supervised Generative Adversarial Networks for 3D Reconstruction  [[pdf]](https://arxiv.org/abs/1705.10904)|2017|5
|205|Auto-Encoder Guided GAN for Chinese Calligraphy Synthesis  [[pdf]](https://arxiv.org/abs/1706.08789)|2017|5
|206|ExprGAN: Facial Expression Editing with Controllable Expression Intensity [[pdf]](https://arxiv.org/pdf/1709.03842.pdf)|2017|5
|207| Learning Texture Manifolds with the Periodic Spatial GAN [[pdf]](https://arxiv.org/abs/1705.06566)|2017|5
|208|SAD-GAN: Synthetic Autonomous Driving using Generative Adversarial Networks    [[pdf]](https://arxiv.org/pdf/1611.08788.pdf)|2017|5
|209|Compressed Sensing MRI Reconstruction with Cyclic Loss in Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.00753.pdf)|2017|5
|210|Generative Adversarial Network-based Synthesis of Visible Faces from Polarimetric Thermal Faces [[pdf]](https://arxiv.org/pdf/1708.02681.pdf)|2017|5
|211|Guiding InfoGAN with Semi-Supervision [[pdf]](https://arxiv.org/pdf/1707.04487.pdf)|2017|5
|212|APE-GAN: Adversarial Perturbation Elimination with GAN [[pdf]](https://arxiv.org/pdf/1707.05474.pdf)|2017|5
|213|Binary Generative Adversarial Networks for Image Retrieval [[pdf]](https://arxiv.org/pdf/1708.04150.pdf)|2017|5
|214|Message Passing Multi-Agent GANs (MPM-GAN)  [[pdf]](https://arxiv.org/pdf/1612.01294.pdf)|2017|4
|215|Multi-view Generative Adversarial Networks (MV-BiGAN)  [[pdf]](https://arxiv.org/pdf/1611.02019.pdf)|2017|4
|216|MARTA GANs: Unsupervised Representation Learning for Remote Sensing Image Classification [[pdf]](https://arxiv.org/pdf/1612.08879.pdf)|2017|4
|217|Associative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1611.06953)|2017|4
|218|Generative Adversarial Networks for Multimodal Representation Learning in Video Hyperlinking [[pdf]](https://arxiv.org/abs/1705.05103)|2017|4
|219|TextureGAN: Controlling Deep Image Synthesis with Texture Patches  [[pdf]](https://arxiv.org/abs/1706.02823)|2017|4
|220|On the effect of Batch Normalization and Weight Normalization in Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1704.03971)|2017|4
|221|Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN  [[pdf]](https://arxiv.org/pdf/1705.03387)|2017|4
|222|SeGAN: Segmenting and Generating the Invisible   [[pdf]](https://arxiv.org/pdf/1703.10239.pdf)|2017|3
|223|Softmax GAN [[pdf]](https://arxiv.org/pdf/1704.06191)|2017|3
|224|Class-Splitting Generative Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.07359.pdf)|2017|3
|225|High-Quality Facial Photo-Sketch Synthesis Using Multi-Adversarial Networks [[pdf]](https://arxiv.org/pdf/1710.10182.pdf)|2017|3
|226|Generative Adversarial Structured Networks  [[pdf]](https://sites.google.com/site/nips2016adversarial/WAT16_paper_14.pdf)|2017|3
|227|Geometric GAN  [[pdf]](https://arxiv.org/pdf/1705.02894.pdf)|2017|3
|228|Dualing GANs [[pdf]](https://arxiv.org/pdf/1706.06216.pdf)|2017|3
|229|Adversarial nets with perceptual losses for text-to-image synthesis [[pdf]](https://arxiv.org/pdf/1708.09321.pdf)|2017|3
|230|ARIGAN: Synthetic Arabidopsis Plants using Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1709.00938.pdf)|2017|3
|231|GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks [[pdf]](https://arxiv.org/pdf/1710.00962.pdf)|2017|3
|232|How to Fool Radiologists with Generative Adversarial Networks? A Visual Turing Test for Lung Cancer Diagnosis [[pdf]](https://arxiv.org/pdf/1710.09762.pdf)|2017|3
|233|Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1709.07592.pdf)|2017|3
|234|Sharpness-aware Low dose CT denoising using conditional generative adversarial network [[pdf]](https://arxiv.org/pdf/1708.06453.pdf)|2017|3
|235|AlignGAN: Learning to Align Cross-Domain Images with Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1707.01400.pdf)|2017|2
|236|Generate Identity-Preserving Faces by Generative Adversarial Networks  [[pdf]](https://arxiv.org/abs/1706.03227)|2017|2
|237|Creatism: A deep-learning photographer capable of creating professional work [[pdf]](https://arxiv.org/pdf/1707.03491.pdf%20)|2017|2
|238|Image Generation and Editing with Variational Info Generative Adversarial Networks (ViGAN)  [[pdf]](https://arxiv.org/pdf/1701.04568.pdf)|2016|2
|239|Supervised Adversarial Networks for Image Saliency Detection [[pdf]](https://arxiv.org/pdf/1704.07242)|2017|2
|240|Training Triplet Networks with GAN  [[pdf]](https://arxiv.org/pdf/1704.02227)|2017|2
|241|Training Triplet Networks with GAN  [[pdf]](https://arxiv.org/pdf/1704.02227)|2017|2
|242|Generative Mixture of Networks  [[pdf]](https://arxiv.org/pdf/1702.03307.pdf)|2017|2
|243|Stopping GAN Violence: Generative Unadversarial Networks  [[pdf]](https://arxiv.org/pdf/1703.02528.pdf)|2016|2
|244|A Classification-Based Perspective on GAN Distributions [[pdf]](https://arxiv.org/pdf/1711.00970.pdf)|2017|2
|245|CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning [[pdf]](https://arxiv.org/pdf/1710.05106.pdf)|2017|2
|246|Statistics of Deep Generated Images [[pdf]](https://arxiv.org/pdf/1708.02688.pdf)|2017|2
|247|Anti-Makeup: Learning A Bi-Level Adversarial Network for Makeup-Invariant Face Verification [[pdf]](https://arxiv.org/pdf/1709.03654.pdf)|2017|2
|248|Freehand Ultrasound Image Simulation with Spatially-Conditioned Generative Adversarial Networks [[pdf]](https://arxiv.org/ftp/arxiv/papers/1707/1707.05392.pdf)|2017|2
|249|Simultaneously Color-Depth Super-Resolution with Conditional Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1708.09105.pdf)|2017|2
|250|Socially-compliant Navigation through Raw Depth Inputs with Generative Adversarial Imitation Learning [[pdf]](https://arxiv.org/pdf/1710.02543.pdf)|2017|2
|251|Adversarial Generation of Training Examples for Vehicle License Plate Recognition [[pdf]](https://arxiv.org/pdf/1707.03124.pdf)|2017|1
|252|Aesthetic-Driven Image Enhancement by Adversarial Learning [[pdf]](https://arxiv.org/pdf/1707.05251.pdf)|2017|1
|253|Generative Adversarial Models for People Attribute Recognition in Surveillance [[pdf]](https://arxiv.org/pdf/1707.02240.pdf)|2017|1
|254|Improving Heterogeneous Face Recognition with Conditional Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.02848.pdf)|2017|1
|255|Intraoperative Organ Motion Models with an Ensemble of Conditional Generative Adversarial Networks  [[pdf]](https://arxiv.org/ftp/arxiv/papers/1709/1709.02255.pdf)|2017|1
|256|Label Denoising Adversarial Network (LDAN) for Inverse Lighting of Face Images [[pdf]](https://arxiv.org/pdf/1709.01993.pdf)|2017|1
|257|Face Super-Resolution Through Wasserstein GANs  [[pdf]](https://arxiv.org/pdf/1705.02438)|2017|1
|258|Continual Learning in Generative Adversarial Nets [[pdf]](https://arxiv.org/abs/1705.08395)|2017|1
|259|An Adversarial Regularisation for Semi-Supervised Training of Structured Output Neural Networks  [[pdf]](https://arxiv.org/pdf/1702.02382)|2017|1
|260|Generative Cooperative Net for Image Generation and Data Augmentation  [[pdf]](https://arxiv.org/pdf/1705.02887.pdf)|2017|1
|261|Learning Loss for Knowledge Distillation with Conditional Adversarial Networks  [[pdf]](https://arxiv.org/pdf/1709.00513.pdf)|2017|1
|262|Parametrizing filters of a CNN with a GAN [[pdf]](https://arxiv.org/pdf/1710.11386.pdf)|2017|1
|263|A step towards procedural terrain generation with GANs [[pdf]](https://arxiv.org/pdf/1707.03383.pdf)|2017|1
|264|Adversarial Networks for Spatial Context-Aware Spectral Image Reconstruction from RGB [[pdf]](https://arxiv.org/pdf/1709.00265.pdf)|2017|1
|265|Controllable Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1708.00598.pdf)|2017|1
|266|Learning a Generative Adversarial Network for High Resolution Artwork Synthesis [[pdf]](https://arxiv.org/pdf/1708.09533.pdf)|2017|1
|267|Microscopy Cell Segmentation via Adversarial Neural Networks  [[pdf]](https://arxiv.org/pdf/1709.05860.pdf)|2017|1
|268|Neural Stain-Style Transfer Learning using GAN for Histopathological Images [[pdf]](https://arxiv.org/pdf/1710.08543.pdf)|2017|1
|269|Deep and Hierarchical Implicit Models (Bayesian GAN)  [[pdf]](https://arxiv.org/pdf/1702.08896.pdf)|2017|0
|270|How to Train Your DRAGAN [[pdf]](https://arxiv.org/abs/1705.07215)|2017|0
|271|Activation Maximization Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1703.02000.pdf)|2017|0
|272|Generative Adversarial Nets with Labeled Data by Activation Maximization (AMGAN)  [[pdf]](https://arxiv.org/pdf/1703.02000.pdf)|2017|0
|273|Depth Structure Preserving Scene Image Generation  [[pdf]](https://arxiv.org/abs/1706.00212)|2017|0
|274|Synthesizing Filamentary Structured Images with GANs  [[pdf]](https://arxiv.org/abs/1706.02185)|2017|0
|275|Bayesian Conditional Generative Adverserial Networks [[pdf]](https://arxiv.org/pdf/1706.05477.pdf)|2017|0
|276|Generative Adversarial Networks with Inverse Transformation Unit [[pdf]](https://arxiv.org/pdf/1709.09354.pdf)|2017|0
|277|Image Quality Assessment Techniques Show Improved Training and Evaluation of Autoencoder Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1708.02237.pdf)|2017|0
|278|KGAN: How to Break The Minimax Game in GAN  [[pdf]](https://arxiv.org/pdf/1711.01744.pdf)|2017|0
|279|Linking Generative Adversarial Learning and Binary Classification [[pdf]](https://arxiv.org/pdf/1709.01509.pdf)|2017|0
|280|Structured Generative Adversarial Networks  [[pdf]](http://papers.nips.cc/paper/6979-structured-generative-adversarial-networks.pdf)|2017|0
|281|Tensorizing Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1710.10772.pdf)|2017|0
|282|A Novel Approach to Artistic Textual Visualization via GAN [[pdf]](https://arxiv.org/pdf/1710.10553.pdf)|2017|0
|283|Artificial Generation of Big Data for Improving Image Classification: A Generative Adversarial Network Approach on SAR Data [[pdf]](https://arxiv.org/pdf/1711.02010.pdf)|2017|0
|284|Conditional Adversarial Network for Semantic Segmentation of Brain Tumor [[pdf]](https://arxiv.org/pdf/1708.05227)|2017|0
|285|Data Augmentation in Classification using GAN [[pdf]](https://arxiv.org/pdf/1711.00648.pdf)|2017|0
|286|Deep Generative Adversarial Neural Networks for Realistic Prostate Lesion MRI Synthesis [[pdf]](https://arxiv.org/pdf/1708.00129.pdf)|2017|0
|287|Face Transfer with Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1710.06090.pdf)|2017|0
|288|Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets  [[pdf]](https://arxiv.org/pdf/1710.04835.pdf)|2017|0
|289|Generative Adversarial Network based on Resnet for Conditional Image Restoration  [[pdf]](https://arxiv.org/pdf/1707.04881.pdf)|2017|0
|290|Hierarchical Detail Enhancing Mesh-Based Shape Generation with 3D Generative Adversarial Network [[pdf]](https://arxiv.org/pdf/1709.07581.pdf)|2017|0
|291|High-Quality Face Image SR Using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1707.00737.pdf)|2017|0
|292|Improving image generative models with human interactions [[pdf]](https://arxiv.org/pdf/1709.10459.pdf)|2017|0
|293|Learning to Generate Chairs with Generative Adversarial Nets [[pdf]](https://arxiv.org/pdf/1705.10413.pdf)|2017|0
|294|Retinal Vasculature Segmentation Using Local Saliency Maps and Generative Adversarial Networks For Image Super Resolution [[pdf]](https://arxiv.org/pdf/1710.04783.pdf)|2017|0



----------

### :notebook_with_decorative_cover: Theory
- Improved Techniques for Training GANs [[pdf]](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans.pdf) 	
- Energy-Based GANs & other Adversarial things by Yann Le Cun [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- Mode RegularizedGenerative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1612.02136.pdf)

----------

### :nut_and_bolt: Presentations
- Generative Adversarial Networks (GANs) by Ian Goodfellow [[pdf]](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf) 	
- Learning Deep Generative Models by Russ Salakhutdinov [[pdf]](http://www.cs.toronto.edu/~rsalakhu/talk_Montreal_2016_Salakhutdinov.pdf) 	

----------
### :books: Courses / Tutorials / Blogs (Webpages unless other is stated)
- NIPS 2016 Tutorial: Generative Adversarial Networks (2016) [[pdf]](https://arxiv.org/pdf/1701.00160.pdf)
- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
- [Generative Models by OpenAI](https://openai.com/blog/generative-models/)
- [MNIST Generative Adversarial Model in Keras](https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/)
- [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/)
- [Attacking machine learning with adversarial examples by OpenAI](https://openai.com/blog/adversarial-example-research/)
- [On the intuition behind deep learning & GANs—towards a fundamental understanding](https://blog.waya.ai/introduction-to-gans-a-boxing-match-b-w-neural-nets-b4e5319cc935)
- [SimGANs - a game changer in unsupervised learning, self driving cars, and more](https://blog.waya.ai/simgans-applied-to-autonomous-driving-5a8c6676e36b)
----------

### :package: Resources / Models (Descending order based on GitHub stars)
|S/N|Name|Repo|Stars
:---:|:---:|:---:|:---:
|1|Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)|https://github.com/junyanz/CycleGAN|6383
|2| :chart_with_upwards_trend: Image-to-image translation with conditional adversarial nets (pix2pix)|https://github.com/phillipi/pix2pix|4851
|3|Image super-resolution through deep learning|https://github.com/david-gpu/srez|4801
|4|Tensorflow implementation of Deep Convolutional Generative Adversarial Networks (DCGAN)|https://github.com/carpedm20/DCGAN-tensorflow|4223
|5| :chart_with_upwards_trend: Generative Models: Collection of generative models, e.g. GAN, VAE in Pytorch and Tensorflow|https://github.com/wiseodd/generative-models|3374
|6| :chart_with_upwards_trend: Generative Visual Manipulation on the Natural Image Manifold (iGAN)|https://github.com/junyanz/iGAN|2775
|7|Deep Convolutional Generative Adversarial Networks (DCGAN)|https://github.com/Newmu/dcgan_code|2750
|8| :chart_with_upwards_trend: cleverhans: A library for benchmarking vulnerability to adversarial examples|https://github.com/openai/cleverhans|1939"
|9| :chart_with_upwards_trend: Wasserstein GAN|https://github.com/martinarjovsky/WassersteinGAN|1784
|10|Neural Photo Editing with Introspective Adversarial Networks|https://github.com/ajbrock/Neural-Photo-Editor|1709
|11|Generative Adversarial Text to Image Synthesis |https://github.com/paarthneekhara/text-to-image|1648
|12|Improved Techniques for Training GANs|https://github.com/openai/improved-gan|1345
|13 :chart_with_upwards_trend: |Improved Training of Wasserstein GANs|https://github.com/igul222/improved_wgan_training|1114
|14|StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks|https://github.com/hanzhanggit/StackGAN|1091
|15|Semantic Image Inpainting with Perceptual and Contextual Losses (2016) |https://github.com/bamos/dcgan-completion.tensorflow|998
|16|HyperGAN|https://github.com/255bits/HyperGAN|759
|17| :chart_with_upwards_trend: Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)|https://github.com/devnag/pytorch-generative-adversarial-networks|758
|18| :chart_with_upwards_trend: Learning to Discover Cross-Domain Relations with Generative Adversarial Networks|https://github.com/carpedm20/DiscoGAN-pytorch|710
|19|Unsupervised Cross-Domain Image Generation|https://github.com/yunjey/domain-transfer-network|652
|20|Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (KERAS-DCGAN)|https://github.com/jacobgil/keras-dcgan|649
|21|Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks (The Eyescream Project)|https://github.com/facebook/eyescream|566
|22| :chart_with_upwards_trend: Image-to-image translation using conditional adversarial nets|https://github.com/yenchenlin/pix2pix-tensorflow|548
|23|Generating Videos with Scene Dynamics|https://github.com/cvondrick/videogan|537
|24|Deep multi-scale video prediction beyond mean square error|https://github.com/dyelax/Adversarial_Video_Generation|454
|25|Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space|https://github.com/Evolving-AI-Lab/ppgn|450
|26|Learning from Simulated and Unsupervised Images through Adversarial Training|https://github.com/carpedm20/simulated-unsupervised-tensorflow|448
|27|Synthesizing the preferred inputs for neurons in neural networks via deep generator networks|https://github.com/Evolving-AI-Lab/synthesizing|421
|28| :chart_with_upwards_trend: Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling|https://github.com/zck119/3dgan-release|383
|29|A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection|https://github.com/xiaolonw/adversarial-frcnn|362
|30|Conditional Image Synthesis With Auxiliary Classifier GANs|https://github.com/buriburisuri/ac-gan|312
|31|Generating images with recurrent adversarial networks (sequence_gan)|https://github.com/ofirnachum/sequence_gan|298
|32|Learning What and Where to Draw|https://github.com/reedscot/nips2016|294
|33|Adversarially Learned Inference (2016) (ALI)|https://github.com/IshmaelBelghazi/ALI|248
|34|Precomputed real-time texture synthesis with markovian generative adversarial networks|https://github.com/chuanli11/MGANs|235
|35|Autoencoding beyond pixels using a learned similarity metric|https://github.com/andersbll/autoencoding_beyond_pixels|235
|36|Unrolled Generative Adversarial Networks|https://github.com/poolio/unrolled_gan|223
|37|Sampling Generative Networks|https://github.com/dribnet/plat|198
|38|Energy-based generative adversarial network|https://github.com/buriburisuri/ebgan|194
|39|Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network|https://github.com/leehomyc/Photo-Realistic-Super-Resoluton|175
|40|Invertible Conditional GANs for image editing|https://github.com/Guim3/IcGAN|174
|41|Pixel-Level Domain Transfer|https://github.com/fxia22/PixelDTGAN|169
|42|SalGAN: Visual Saliency Prediction with Generative Adversarial Networks|https://github.com/imatge-upc/saliency-salgan-2017|164
|43|Generative face completion (2017)|https://github.com/Yijunmaverick/GenerativeFaceCompletion|163
|44| :chart_with_upwards_trend: C-RNN-GAN: Continuous recurrent neural networks with adversarial training|https://github.com/olofmogren/c-rnn-gan|151
|45|Adversarial Autoencoders |https://github.com/musyoku/adversarial-autoencoder|148
|46|Coupled Generative Adversarial Networks|https://github.com/mingyuliutw/CoGAN|148
|47|Context Encoders: Feature Learning by Inpainting (2016)|https://github.com/jazzsaxmafia/Inpainting|108
|48|Generative Image Modeling using Style and Structure Adversarial Networks (ss-gan)|https://github.com/xiaolonw/ss-gan|100
|49|Conditional Generative Adversarial Nets|https://github.com/zhangqianhui/Conditional-Gans|98
|50|InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets|https://github.com/buriburisuri/supervised_infogan|79
|51|Reconstruction of three-dimensional porous media using generative adversarial neural networks |https://github.com/LukasMosser/PorousMediaGan|29
|52|Improving Generative Adversarial Networks with Denoising Feature Matching|https://github.com/hvy/chainer-gan-denoising-feature-matching|15
|53|Least Squares Generative Adversarial Networks|https://github.com/pfnet-research/chainer-LSGAN|9


* 3D-ED-GAN - [Shape Inpainting using 3D Generative Adversarial Network and Recurrent Convolutional Networks](https://arxiv.org/abs/1711.06375) 
* 3D-GAN - [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://arxiv.org/abs/1610.07584) ([github](https://github.com/zck119/3dgan-release))
* 3D-IWGAN - [Improved Adversarial Systems for 3D Object Generation and Reconstruction](https://arxiv.org/abs/1707.09557) ([github](https://github.com/EdwardSmith1884/3D-IWGAN))
* 3D-PhysNet - [3D-PhysNet: Learning the Intuitive Physics of Non-Rigid Object Deformations](https://arxiv.org/abs/1805.00328) 
* 3D-RecGAN - [3D Object Reconstruction from a Single Depth View with Adversarial Learning](https://arxiv.org/abs/1708.07969) ([github](https://github.com/Yang7879/3D-RecGAN))
* ABC-GAN - [ABC-GAN: Adaptive Blur and Control for improved training stability of Generative Adversarial Networks](https://drive.google.com/file/d/0B3wEP_lEl0laVTdGcHE2VnRiMlE/view) ([github](https://github.com/IgorSusmelj/ABC-GAN))
* ABC-GAN - [GANs for LIFE: Generative Adversarial Networks for Likelihood Free Inference](https://arxiv.org/abs/1711.11139) 
* AC-GAN - [Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585) 
* acGAN - [Face Aging With Conditional Generative Adversarial Networks](https://arxiv.org/abs/1702.01983) 
* ACGAN - [Coverless Information Hiding Based on Generative adversarial networks](https://arxiv.org/abs/1712.06951) 
* acGAN - [On-line Adaptative Curriculum Learning for GANs](https://arxiv.org/abs/1808.00020) 
* ACtuAL - [ACtuAL: Actor-Critic Under Adversarial Learning](https://arxiv.org/abs/1711.04755) 
* AdaGAN - [AdaGAN: Boosting Generative Models](https://arxiv.org/abs/1701.02386v1) 
* Adaptive GAN - [Customizing an Adversarial Example Generator with Class-Conditional GANs](https://arxiv.org/abs/1806.10496) 
* AdvEntuRe - [AdvEntuRe: Adversarial Training for Textual Entailment with Knowledge-Guided Examples](https://arxiv.org/abs/1805.04680) 
* AdvGAN - [Generating adversarial examples with adversarial networks](https://arxiv.org/abs/1801.02610) 
* AE-GAN - [AE-GAN: adversarial eliminating with GAN](https://arxiv.org/abs/1707.05474) 
* AE-OT - [Latent Space Optimal Transport for Generative Models](https://arxiv.org/abs/1809.05964) 
* AEGAN - [Learning Inverse Mapping by Autoencoder based Generative Adversarial Nets](https://arxiv.org/abs/1703.10094) 
* AF-DCGAN - [AF-DCGAN: Amplitude Feature Deep Convolutional GAN for Fingerprint Construction in Indoor Localization System](https://arxiv.org/abs/1804.05347) 
* AffGAN - [Amortised MAP Inference for Image Super-resolution](https://arxiv.org/abs/1610.04490) 
* AIM - [Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization](https://arxiv.org/abs/1809.05972) 
* AL-CGAN - [Learning to Generate Images of Outdoor Scenes from Attributes and Semantic Layouts](https://arxiv.org/abs/1612.00215) 
* ALI - [Adversarially Learned Inference](https://arxiv.org/abs/1606.00704) ([github](https://github.com/IshmaelBelghazi/ALI))
* AlignGAN - [AlignGAN: Learning to Align Cross-Domain Images with Conditional Generative Adversarial Networks](https://arxiv.org/abs/1707.01400) 
* AlphaGAN - [AlphaGAN: Generative adversarial networks for natural image matting](https://arxiv.org/abs/1807.10088) 
* AM-GAN - [Activation Maximization Generative Adversarial Nets](https://arxiv.org/abs/1703.02000) 
* AmbientGAN - [AmbientGAN: Generative models from lossy measurements](https://openreview.net/forum?id=Hy7fDog0b) ([github](https://github.com/AshishBora/ambient-gan))
* AMC-GAN - [Video Prediction with Appearance and Motion Conditions](https://arxiv.org/abs/1807.02635) 
* AnoGAN - [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://arxiv.org/abs/1703.05921v1) 
* APD - [Adversarial Distillation of Bayesian Neural Network Posteriors](https://arxiv.org/abs/1806.10317) 
* APE-GAN - [APE-GAN: Adversarial Perturbation Elimination with GAN](https://arxiv.org/abs/1707.05474) 
* ARAE - [Adversarially Regularized Autoencoders for Generating Discrete Structures](https://arxiv.org/abs/1706.04223) ([github](https://github.com/jakezhaojb/ARAE))
* ARDA - [Adversarial Representation Learning for Domain Adaptation](https://arxiv.org/abs/1707.01217) 
* ARIGAN - [ARIGAN: Synthetic Arabidopsis Plants using Generative Adversarial Network](https://arxiv.org/abs/1709.00938) 
* ArtGAN - [ArtGAN: Artwork Synthesis with Conditional Categorial GANs](https://arxiv.org/abs/1702.03410) 
* ASDL-GAN - [Automatic Steganographic Distortion Learning Using a Generative Adversarial Network](https://ieeexplore.ieee.org/document/8017430/) 
* ATA-GAN - [Attention-Aware Generative Adversarial Networks (ATA-GANs)](https://arxiv.org/abs/1802.09070) 
* Attention-GAN - [Attention-GAN for Object Transfiguration in Wild Images](https://arxiv.org/abs/1803.06798) 
* AttGAN - [Arbitrary Facial Attribute Editing: Only Change What You Want](https://arxiv.org/abs/1711.10678) ([github](https://github.com/LynnHo/AttGAN-Tensorflow))
* AttnGAN - [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://arxiv.org/abs/1711.10485) ([github](https://github.com/taoxugit/AttnGAN))
* AVID - [AVID: Adversarial Visual Irregularity Detection](https://arxiv.org/abs/1805.09521) 
* B-DCGAN - [B-DCGAN:Evaluation of Binarized DCGAN for FPGA](https://arxiv.org/abs/1803.10930) 
* b-GAN - [Generative Adversarial Nets from a Density Ratio Estimation Perspective](https://arxiv.org/abs/1610.02920) 
* BAGAN - [BAGAN: Data Augmentation with Balancing GAN](https://arxiv.org/abs/1803.09655) 
* Bayesian GAN - [Deep and Hierarchical Implicit Models](https://arxiv.org/abs/1702.08896) 
* Bayesian GAN - [Bayesian GAN](https://arxiv.org/abs/1705.09558) ([github](https://github.com/andrewgordonwilson/bayesgan/))
* BCGAN - [Bayesian Conditional Generative Adverserial Networks](https://arxiv.org/abs/1706.05477) 
* BCGAN - [Bidirectional Conditional Generative Adversarial networks](https://arxiv.org/abs/1711.07461) 
* BEAM - [Boltzmann Encoded Adversarial Machines](https://arxiv.org/abs/1804.08682) 
* BEGAN - [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717) 
* BEGAN-CS - [Escaping from Collapsing Modes in a Constrained Space](https://arxiv.org/abs/1808.07258) 
* Bellman GAN - [Distributional Multivariate Policy Evaluation and Exploration with the Bellman GAN](https://arxiv.org/abs/1808.01960) 
* BGAN - [Binary Generative Adversarial Networks for Image Retrieval](https://arxiv.org/abs/1708.04150) ([github](https://github.com/htconquer/BGAN))
* Bi-GAN - [Autonomously and Simultaneously Refining Deep Neural Network Parameters by a Bi-Generative Adversarial Network Aided Genetic Algorithm](https://arxiv.org/abs/1809.10244) 
* BicycleGAN - [Toward Multimodal Image-to-Image Translation](https://arxiv.org/abs/1711.11586) ([github](https://github.com/junyanz/BicycleGAN))
* BiGAN - [Adversarial Feature Learning](https://arxiv.org/abs/1605.09782v7) 
* BinGAN - [BinGAN: Learning Compact Binary Descriptors with a Regularized GAN](https://arxiv.org/abs/1806.06778) 
* BourGAN - [BourGAN: Generative Networks with Metric Embeddings](https://arxiv.org/abs/1805.07674) 
* BranchGAN - [Branched Generative Adversarial Networks for Multi-Scale Image Manifold Learning](https://arxiv.org/abs/1803.08467) 
* BRE - [Improving GAN Training via Binarized Representation Entropy (BRE) Regularization](https://arxiv.org/abs/1805.03644) ([github](https://github.com/BorealisAI/bre-gan))
* BridgeGAN - [Generative Adversarial Frontal View to Bird View Synthesis](https://arxiv.org/abs/1808.00327) 
* BS-GAN - [Boundary-Seeking Generative Adversarial Networks](https://arxiv.org/abs/1702.08431v1) 
* BubGAN - [BubGAN: Bubble Generative Adversarial Networks for Synthesizing Realistic Bubbly Flow Images](https://arxiv.org/abs/1809.02266) 
* BWGAN - [Banach Wasserstein GAN](https://arxiv.org/abs/1806.06621) 
* C-GAN  - [Face Aging with Contextual Generative Adversarial Nets ](https://arxiv.org/abs/1802.00237 ) 
* C-RNN-GAN - [C-RNN-GAN: Continuous recurrent neural networks with adversarial training](https://arxiv.org/abs/1611.09904) ([github](https://github.com/olofmogren/c-rnn-gan/))
* CA-GAN - [Composition-aided Sketch-realistic Portrait Generation](https://arxiv.org/abs/1712.00899) 
* CaloGAN - [CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks](https://arxiv.org/abs/1705.02355) ([github](https://github.com/hep-lbdl/CaloGAN))
* CAN - [CAN: Creative Adversarial Networks, Generating Art by Learning About Styles and Deviating from Style Norms](https://arxiv.org/abs/1706.07068) 
* CapsGAN - [CapsGAN: Using Dynamic Routing for Generative Adversarial Networks](https://arxiv.org/abs/1806.03968) 
* CapsuleGAN - [CapsuleGAN: Generative Adversarial Capsule Network ](http://arxiv.org/abs/1802.06167) 
* CatGAN - [Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks](https://arxiv.org/abs/1511.06390v2) 
* CatGAN - [CatGAN: Coupled Adversarial Transfer for Domain Generation](https://arxiv.org/abs/1711.08904) 
* CausalGAN - [CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training](https://arxiv.org/abs/1709.02023) 
* CC-GAN - [Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks](https://arxiv.org/abs/1611.06430) ([github](https://github.com/edenton/cc-gan))
* cd-GAN - [Conditional Image-to-Image Translation](https://arxiv.org/abs/1805.00251) 
* CDcGAN - [Simultaneously Color-Depth Super-Resolution with Conditional Generative Adversarial Network](https://arxiv.org/abs/1708.09105) 
* CE-GAN - [Deep Learning for Imbalance Data Classification using Class Expert Generative Adversarial Network](https://arxiv.org/abs/1807.04585) 
* CFG-GAN - [Composite Functional Gradient Learning of Generative Adversarial Models](https://arxiv.org/abs/1801.06309) 
* CGAN - [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) 
* CGAN - [Controllable Generative Adversarial Network](https://arxiv.org/abs/1708.00598) 
* Chekhov GAN - [An Online Learning Approach to Generative Adversarial Networks](https://arxiv.org/abs/1706.03269) 
* ciGAN - [Conditional Infilling GANs for Data Augmentation in Mammogram Classification](https://arxiv.org/abs/1807.08093) 
* CinCGAN - [Unsupervised Image Super-Resolution using Cycle-in-Cycle Generative Adversarial Networks](https://arxiv.org/abs/1809.00437) 
* CipherGAN - [Unsupervised Cipher Cracking Using Discrete GANs](https://arxiv.org/abs/1801.04883) 
* ClusterGAN - [ClusterGAN : Latent Space Clustering in Generative Adversarial Networks](https://arxiv.org/abs/1809.03627) 
* CM-GAN - [CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning](https://arxiv.org/abs/1710.05106) 
* CoAtt-GAN - [Are You Talking to Me? Reasoned Visual Dialog Generation through Adversarial Learning](https://arxiv.org/abs/1711.07613) 
* CoGAN - [Coupled Generative Adversarial Networks](https://arxiv.org/abs/1606.07536v2) 
* ComboGAN - [ComboGAN: Unrestrained Scalability for Image Domain Translation](https://arxiv.org/abs/1712.06909) ([github](https://github.com/AAnoosheh/ComboGAN))
* ConceptGAN - [Learning Compositional Visual Concepts with Mutual Consistency](https://arxiv.org/abs/1711.06148) 
* Conditional cycleGAN - [Conditional CycleGAN for Attribute Guided Face Image Generation](https://arxiv.org/abs/1705.09966) 
* constrast-GAN - [Generative Semantic Manipulation with Contrasting GAN](https://arxiv.org/abs/1708.00315) 
* Context-RNN-GAN - [Contextual RNN-GANs for Abstract Reasoning Diagram Generation](https://arxiv.org/abs/1609.09444) 
* CorrGAN - [Correlated discrete data generation using adversarial training](https://arxiv.org/abs/1804.00925) 
* Coulomb GAN - [Coulomb GANs: Provably Optimal Nash Equilibria via Potential Fields](https://arxiv.org/abs/1708.08819) 
* Cover-GAN - [Generative Steganography with Kerckhoffs' Principle based on Generative Adversarial Networks](https://arxiv.org/abs/1711.04916) 
* cowboy - [Defending Against Adversarial Attacks by Leveraging an Entire GAN](https://arxiv.org/abs/1805.10652) 
* CR-GAN - [CR-GAN: Learning Complete Representations for Multi-view Generation](https://arxiv.org/abs/1806.11191) 
* Cramèr GAN  - [The Cramer Distance as a Solution to Biased Wasserstein Gradients](https://arxiv.org/abs/1705.10743) 
* Cross-GAN - [Crossing Generative Adversarial Networks for Cross-View Person Re-identification](https://arxiv.org/abs/1801.01760) 
* crVAE-GAN - [Channel-Recurrent Variational Autoencoders](https://arxiv.org/abs/1706.03729) 
* CS-GAN - [Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets](https://arxiv.org/abs/1703.04887) 
* CSG - [Speech-Driven Expressive Talking Lips with Conditional Sequential Generative Adversarial Networks](https://arxiv.org/abs/1806.00154) 
* CT-GAN - [CT-GAN: Conditional Transformation Generative Adversarial Network for Image Attribute Modification](https://arxiv.org/abs/1807.04812) 
* CVAE-GAN - [CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training](https://arxiv.org/abs/1703.10155) 
* CycleGAN - [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) ([github](https://github.com/junyanz/CycleGAN))
* D-GAN - [Differential Generative Adversarial Networks: Synthesizing Non-linear Facial Variations with Limited Number of Training Data](https://arxiv.org/abs/1711.10267) 
* D-WCGAN - [I-vector Transformation Using Conditional Generative Adversarial Networks for Short Utterance Speaker Verification](https://arxiv.org/abs/1804.00290) 
* D2GAN - [Dual Discriminator Generative Adversarial Nets](http://arxiv.org/abs/1709.03831) 
* D2IA-GAN - [Tagging like Humans: Diverse and Distinct Image Annotation](https://arxiv.org/abs/1804.00113) 
* DA-GAN  - [DA-GAN: Instance-level Image Translation by Deep Attention Generative Adversarial Networks (with Supplementary Materials)](http://arxiv.org/abs/1802.06454) 
* DADA - [DADA: Deep Adversarial Data Augmentation for Extremely Low Data Regime Classification](https://arxiv.org/abs/1809.00981) 
* DAGAN - [Data Augmentation Generative Adversarial Networks](https://arxiv.org/abs/1711.04340) 
* DAN - [Distributional Adversarial Networks](https://arxiv.org/abs/1706.09549) 
* DBLRGAN - [Adversarial Spatio-Temporal Learning for Video Deblurring](https://arxiv.org/abs/1804.00533) 
* DCGAN - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) ([github](https://github.com/Newmu/dcgan_code))
* DE-GAN - [Generative Adversarial Networks with Decoder-Encoder Output Noise](https://arxiv.org/abs/1807.03923) 
* DeblurGAN - [DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks](https://arxiv.org/abs/1711.07064) ([github](https://github.com/KupynOrest/DeblurGAN))
* DeepFD - [Learning to Detect Fake Face Images in the Wild](https://arxiv.org/abs/1809.08754) 
* Defense-GAN - [Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models ](https://arxiv.org/abs/1805.06605 ) ([github](https://github.com/kabkabm/defensegan))
* Defo-Net - [Defo-Net: Learning Body Deformation using Generative Adversarial Networks](https://arxiv.org/abs/1804.05928) 
* DeliGAN - [DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data](https://arxiv.org/abs/1706.02071) ([github](https://github.com/val-iisc/deligan))
* DF-GAN - [Learning Disentangling and Fusing Networks for Face Completion Under Structured Occlusions](https://arxiv.org/abs/1712.04646) 
* DialogWAE - [DialogWAE: Multimodal Response Generation with Conditional Wasserstein Auto-Encoder](https://arxiv.org/abs/1805.12352) 
* DiscoGAN - [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192v1) 
* DistanceGAN - [One-Sided Unsupervised Domain Mapping](https://arxiv.org/abs/1706.00826) 
* DM-GAN - [Dual Motion GAN for Future-Flow Embedded Video Prediction](https://arxiv.org/abs/1708.00284) 
* DMGAN - [Disconnected Manifold Learning for Generative Adversarial Networks](https://arxiv.org/abs/1806.00880) 
* DNA-GAN - [DNA-GAN: Learning Disentangled Representations from Multi-Attribute Images](https://arxiv.org/abs/1711.05415) 
* DOPING - [DOPING: Generative Data Augmentation for Unsupervised Anomaly Detection with GAN](https://arxiv.org/abs/1808.07632) 
* dp-GAN - [Differentially Private Releasing via Deep Generative Model](https://arxiv.org/abs/1801.01594) 
* DP-GAN - [DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text ](https://arxiv.org/abs/1802.01345 ) 
* DPGAN  - [Differentially Private Generative Adversarial Network ](http://arxiv.org/abs/1802.06739) 
* DR-GAN - [Representation Learning by Rotating Your Faces](https://arxiv.org/abs/1705.11136) 
* DRAGAN - [How to Train Your DRAGAN](https://arxiv.org/abs/1705.07215) ([github](https://github.com/kodalinaveen3/DRAGAN))
* Dropout-GAN - [Dropout-GAN: Learning from a Dynamic Ensemble of Discriminators](https://arxiv.org/abs/1807.11346) 
* DRPAN - [Discriminative Region Proposal Adversarial Networks for High-Quality Image-to-Image Translation](https://arxiv.org/abs/1711.09554) 
* DSH-GAN - [Deep Semantic Hashing with Generative Adversarial Networks](https://arxiv.org/abs/1804.08275) 
* DSP-GAN - [Depth Structure Preserving Scene Image Generation](https://arxiv.org/abs/1706.00212) 
* DTLC-GAN - [Generative Adversarial Image Synthesis with Decision Tree Latent Controller](https://arxiv.org/abs/1805.10603) 
* DTN - [Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200) 
* DTR-GAN - [DTR-GAN: Dilated Temporal Relational Adversarial Network for Video Summarization](https://arxiv.org/abs/1804.11228) 
* DualGAN - [DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/abs/1704.02510v1) 
* Dualing GAN - [Dualing GANs](https://arxiv.org/abs/1706.06216) 
* DVGAN - [Human Motion Modeling using DVGANs](https://arxiv.org/abs/1804.10652) 
* Dynamics Transfer GAN - [Dynamics Transfer GAN: Generating Video by Transferring Arbitrary Temporal Dynamics from a Source Video to a Single Target Image](https://arxiv.org/abs/1712.03534) 
* E-GAN - [Evolutionary Generative Adversarial Networks](https://arxiv.org/abs/1803.00657) 
* EAR - [Generative Model for Heterogeneous Inference](https://arxiv.org/abs/1804.09858) 
* EBGAN - [Energy-based Generative Adversarial Network](https://arxiv.org/abs/1609.03126v4) 
* ecGAN - [eCommerceGAN : A Generative Adversarial Network for E-commerce](https://arxiv.org/abs/1801.03244) 
* ED//GAN - [Stabilizing Training of Generative Adversarial Networks through Regularization](https://arxiv.org/abs/1705.09367) 
* Editable GAN - [Editable Generative Adversarial Networks: Generating and Editing Faces Simultaneously](https://arxiv.org/abs/1807.07700) 
* EGAN - [Enhanced Experience Replay Generation for Efficient Reinforcement Learning](https://arxiv.org/abs/1705.08245) 
* EL-GAN - [EL-GAN: Embedding Loss Driven Generative Adversarial Networks for Lane Detection](https://arxiv.org/abs/1806.05525) 
* ELEGANT - [ELEGANT: Exchanging Latent Encodings with GAN for Transferring Multiple Face Attributes](https://arxiv.org/abs/1803.10562) 
* EnergyWGAN - [Energy-relaxed Wassertein GANs (EnergyWGAN): Towards More Stable and High Resolution Image Generation](https://arxiv.org/abs/1712.01026) 
* ESRGAN - [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) 
* ExGAN - [Eye In-Painting with Exemplar Generative Adversarial Networks](https://arxiv.org/abs/1712.03999) 
* ExposureGAN - [Exposure: A White-Box Photo Post-Processing Framework](https://arxiv.org/abs/1709.09602) ([github](https://github.com/yuanming-hu/exposure))
* ExprGAN - [ExprGAN: Facial Expression Editing with Controllable Expression Intensity](https://arxiv.org/abs/1709.03842) 
* f-CLSWGAN - [Feature Generating Networks for Zero-Shot Learning](https://arxiv.org/abs/1712.00981) 
* f-GAN - [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/abs/1606.00709) 
* FairGAN - [FairGAN: Fairness-aware Generative Adversarial Networks](https://arxiv.org/abs/1805.11202) 
* Fairness GAN - [Fairness GAN](https://arxiv.org/abs/1805.09910) 
* FakeGAN - [Detecting Deceptive Reviews using Generative Adversarial Networks](https://arxiv.org/abs/1805.10364) 
* FBGAN - [Feedback GAN (FBGAN) for DNA: a Novel Feedback-Loop Architecture for Optimizing Protein Functions](https://arxiv.org/abs/1804.01694) 
* FBGAN - [Featurized Bidirectional GAN: Adversarial Defense via Adversarially Learned Semantic Inference](https://arxiv.org/abs/1805.07862) 
* FC-GAN - [Fast-converging Conditional Generative Adversarial Networks for Image Synthesis](https://arxiv.org/abs/1805.01972) 
* FF-GAN - [Towards Large-Pose Face Frontalization in the Wild](https://arxiv.org/abs/1704.06244) 
* FGGAN - [Adversarial Learning for Fine-grained Image Search](https://arxiv.org/abs/1807.02247) 
* Fictitious GAN - [Fictitious GAN: Training GANs with Historical Models](https://arxiv.org/abs/1803.08647) 
* FIGAN - [Frame Interpolation with Multi-Scale Deep Loss Functions and Generative Adversarial Networks](https://arxiv.org/abs/1711.06045) 
* Fila-GAN - [Synthesizing Filamentary Structured Images with GANs](https://arxiv.org/abs/1706.02185) 
* First Order GAN  - [First Order Generative Adversarial Networks ](https://arxiv.org/abs/1802.04591) ([github](https://github.com/zalandoresearch/first_order_gan))
* Fisher GAN - [Fisher GAN](https://arxiv.org/abs/1705.09675) 
* Flow-GAN - [Flow-GAN: Bridging implicit and prescribed learning in generative models](https://arxiv.org/abs/1705.08868) 
* FrankenGAN - [rankenGAN: Guided Detail Synthesis for Building Mass-Models Using Style-Synchonized GANs](https://arxiv.org/abs/1806.07179) 
* FSEGAN - [Exploring Speech Enhancement with Generative Adversarial Networks for Robust Speech Recognition](https://arxiv.org/abs/1711.05747) 
* FTGAN - [Hierarchical Video Generation from Orthogonal Information: Optical Flow and Texture](https://arxiv.org/abs/1711.09618) 
* FusedGAN - [Semi-supervised FusedGAN for Conditional Image Generation](https://arxiv.org/abs/1801.05551) 
* FusionGAN - [Learning to Fuse Music Genres with Generative Adversarial Dual Learning](https://arxiv.org/abs/1712.01456) 
* FusionGAN - [Generating a Fusion Image: One's Identity and Another's Shape](https://arxiv.org/abs/1804.07455) 
* G2-GAN - [Geometry Guided Adversarial Facial Expression Synthesis](https://arxiv.org/abs/1712.03474) 
* GAAN - [Generative Adversarial Autoencoder Networks](https://arxiv.org/abs/1803.08887) 
* GAF - [Generative Adversarial Forests for Better Conditioned Adversarial Learning](https://arxiv.org/abs/1805.05185) 
* GAGAN - [GAGAN: Geometry-Aware Generative Adverserial Networks](https://arxiv.org/abs/1712.00684) 
* GAIA - [Generative adversarial interpolative autoencoding: adversarial training on latent space interpolations encourage convex latent distributions](https://arxiv.org/abs/1807.06650) 
* GAIN  - [GAIN: Missing Data Imputation using Generative Adversarial Nets](https://arxiv.org/abs/1806.02920) 
* GAMN - [Generative Adversarial Mapping Networks](https://arxiv.org/abs/1709.09820) 
* GAN - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) ([github](https://github.com/goodfeli/adversarial))
* GAN Lab - [GAN Lab: Understanding Complex Deep Generative Models using Interactive Visual Experimentation](https://arxiv.org/abs/1809.01587) 
* GAN Q-learning - [GAN Q-learning](https://arxiv.org/abs/1805.04874) 
* GAN-AD - [Anomaly Detection with Generative Adversarial Networks for Multivariate Time Series](https://arxiv.org/abs/1809.04758) 
* GAN-ATV - [A Novel Approach to Artistic Textual Visualization via GAN](https://arxiv.org/abs/1710.10553) 
* GAN-CLS - [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396) ([github](https://github.com/reedscot/icml2016))
* GAN-RS - [Towards Qualitative Advancement of Underwater Machine Vision with Generative Adversarial Networks](https://arxiv.org/abs/1712.00736) 
* GAN-SD - [Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning](https://arxiv.org/abs/1805.10000) 
* GAN-sep - [GANs for Biological Image Synthesis](https://arxiv.org/abs/1708.04692) ([github](https://github.com/aosokin/biogans))
* GAN-VFS - [Generative Adversarial Network-based Synthesis of Visible Faces from Polarimetric Thermal Faces](https://arxiv.org/abs/1708.02681) 
* GAN-Word2Vec - [Adversarial Training of Word2Vec for Basket Completion](https://arxiv.org/abs/1805.08720) 
* GANAX - [GANAX: A Unified MIMD-SIMD Acceleration for Generative Adversarial Networks](https://arxiv.org/abs/1806.01107) 
* GANCS - [Deep Generative Adversarial Networks for Compressed Sensing Automates MRI](https://arxiv.org/abs/1706.00051) 
* GANDI - [Guiding the search in continuous state-action spaces by learning an action sampling distribution from off-target samples](https://arxiv.org/abs/1711.01391) 
* GANG - [GANGs: Generative Adversarial Network Games](https://arxiv.org/abs/1712.00679) 
* GANG - [Beyond Local Nash Equilibria for Adversarial Networks](https://arxiv.org/abs/1806.07268) 
* GANosaic - [GANosaic: Mosaic Creation with Generative Texture Manifolds](https://arxiv.org/abs/1712.00269) 
* GANVO - [GANVO: Unsupervised Deep Monocular Visual Odometry and Depth Estimation with Generative Adversarial Networks](https://arxiv.org/abs/1809.05786) 
* GAP - [Context-Aware Generative Adversarial Privacy](https://arxiv.org/abs/1710.09549) 
* GAP - [Generative Adversarial Privacy](https://arxiv.org/abs/1807.05306) 
* GATS - [Sample-Efficient Deep RL with Generative Adversarial Tree Search](https://arxiv.org/abs/1806.05780) 
* GAWWN - [Learning What and Where to Draw](https://arxiv.org/abs/1610.02454) ([github](https://github.com/reedscot/nips2016))
* GC-GAN - [Geometry-Contrastive Generative Adversarial Network for Facial Expression Synthesis](https://arxiv.org/abs/1802.01822 ) 
* GcGAN - [Geometry-Consistent Adversarial Networks for One-Sided Unsupervised Domain Mapping](https://arxiv.org/abs/1809.05852) 
* GeneGAN - [GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data](https://arxiv.org/abs/1705.04932) ([github](https://github.com/Prinsphield/GeneGAN))
* GeoGAN - [Generating Instance Segmentation Annotation by Geometry-guided GAN ](https://arxiv.org/abs/1801.08839 ) 
* Geometric GAN - [Geometric GAN](https://arxiv.org/abs/1705.02894) 
* GIN - [Generative Invertible Networks (GIN): Pathophysiology-Interpretable Feature Mapping and Virtual Patient Generation](https://arxiv.org/abs/1808.04495) 
* GLCA-GAN - [Global and Local Consistent Age Generative Adversarial Networks ](https://arxiv.org/abs/1801.08390) 
* GM-GAN - [Gaussian Mixture Generative Adversarial Networks for Diverse Datasets, and the Unsupervised Clustering of Images](https://arxiv.org/abs/1808.10356) 
* GMAN - [Generative Multi-Adversarial Networks](http://arxiv.org/abs/1611.01673) 
* GMM-GAN - [Towards Understanding the Dynamics of Generative Adversarial Networks](https://arxiv.org/abs/1706.09884) 
* GoGAN - [Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking](https://arxiv.org/abs/1704.04865) 
* GONet - [GONet: A Semi-Supervised Deep Learning Approach For Traversability Estimation](https://arxiv.org/abs/1803.03254) 
* GP-GAN - [GP-GAN: Towards Realistic High-Resolution Image Blending](https://arxiv.org/abs/1703.07195) ([github](https://github.com/wuhuikai/GP-GAN))
* GP-GAN - [GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks](https://arxiv.org/abs/1710.00962) 
* GPU - [A generative adversarial framework for positive-unlabeled classification](https://arxiv.org/abs/1711.08054) 
* GRAN - [Generating images with recurrent adversarial networks](https://arxiv.org/abs/1602.05110) ([github](https://github.com/jiwoongim/GRAN))
* Graphical-GAN - [Graphical Generative Adversarial Networks](https://arxiv.org/abs/1804.03429) 
* GraphSGAN - [Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/abs/1809.00130) 
* GraspGAN - [Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping](https://arxiv.org/abs/1709.07857) 
* GT-GAN - [Deep Graph Translation](https://arxiv.org/abs/1805.09980) 
* HAN - [Chinese Typeface Transformation with Hierarchical Adversarial Network](https://arxiv.org/abs/1711.06448) 
* HAN - [Bidirectional Learning for Robust Neural Networks](https://arxiv.org/abs/1805.08006) 
* HiGAN - [Exploiting Images for Video Recognition with Hierarchical Generative Adversarial Networks](https://arxiv.org/abs/1805.04384) 
* HP-GAN - [HP-GAN: Probabilistic 3D human motion prediction via GAN](https://arxiv.org/abs/1711.09561) 
* HR-DCGAN - [High-Resolution Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1711.06491) 
* hredGAN - [Multi-turn Dialogue Response Generation in an Adversarial Learning framework](https://arxiv.org/abs/1805.11752) 
* IAN - [Neural Photo Editing with Introspective Adversarial Networks](https://arxiv.org/abs/1609.07093) ([github](https://github.com/ajbrock/Neural-Photo-Editor))
* IcGAN - [Invertible Conditional GANs for image editing](https://arxiv.org/abs/1611.06355) ([github](https://github.com/Guim3/IcGAN))
* ID-CGAN - [Image De-raining Using a Conditional Generative Adversarial Network](https://arxiv.org/abs/1701.05957v3) 
* IdCycleGAN - [Face Translation between Images and Videos using Identity-aware CycleGAN](https://arxiv.org/abs/1712.00971) 
* IFcVAEGAN - [Conditional Autoencoders with Adversarial Information Factorization](https://arxiv.org/abs/1711.05175) 
* iGAN - [Generative Visual Manipulation on the Natural Image Manifold](https://arxiv.org/abs/1609.03552v2) ([github](https://github.com/junyanz/iGAN))
* IGMM-GAN - [Coupled IGMM-GANs for deep multimodal anomaly detection in human mobility data](https://arxiv.org/abs/1809.02728) 
* Improved GAN - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) ([github](https://github.com/openai/improved-gan))
* In2I - [In2I : Unsupervised Multi-Image-to-Image Translation Using Generative Adversarial Networks](https://arxiv.org/abs/1711.09334) 
* InfoGAN - [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657v1) ([github](https://github.com/openai/InfoGAN))
* IntroVAE - [IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis](https://arxiv.org/abs/1807.06358) 
* IR2VI - [IR2VI: Enhanced Night Environmental Perception by Unsupervised Thermal Image Translation](https://arxiv.org/abs/1806.09565) 
* IRGAN - [IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval models](https://arxiv.org/abs/1705.10513v1) 
* IRGAN - [Generative Adversarial Nets for Information Retrieval: Fundamentals and Advances](https://arxiv.org/abs/1806.03577) 
* ISGAN - [Invisible Steganography via Generative Adversarial Network](https://arxiv.org/abs/1807.08571) 
* ISP-GPM - [Inner Space Preserving Generative Pose Machine](https://arxiv.org/abs/1808.02104) 
* Iterative-GAN - [Two Birds with One Stone: Iteratively Learn Facial Attributes with GANs](https://arxiv.org/abs/1711.06078) ([github](https://github.com/punkcure/Iterative-GAN))
* IterGAN - [IterGANs: Iterative GANs to Learn and Control 3D Object Transformation](https://arxiv.org/abs/1804.05651) 
* IVE-GAN - [IVE-GAN: Invariant Encoding Generative Adversarial Networks](https://arxiv.org/abs/1711.08646) 
* iVGAN - [Towards an Understanding of Our World by GANing Videos in the Wild](https://arxiv.org/abs/1711.11453) ([github](https://github.com/bernhard2202/improved-video-gan))
* IWGAN - [On Unifying Deep Generative Models](https://arxiv.org/abs/1706.00550) 
* JointGAN - [JointGAN: Multi-Domain Joint Distribution Learning with Generative Adversarial Nets](https://arxiv.org/abs/1806.02978) 
* JR-GAN - [JR-GAN: Jacobian Regularization for Generative Adversarial Networks](https://arxiv.org/abs/1806.09235) 
* KBGAN - [KBGAN: Adversarial Learning for Knowledge Graph Embeddings](https://arxiv.org/abs/1711.04071) 
* KGAN - [KGAN: How to Break The Minimax Game in GAN](https://arxiv.org/abs/1711.01744) 
* l-GAN - [Representation Learning and Adversarial Generation of 3D Point Clouds](https://arxiv.org/abs/1707.02392) 
* LAC-GAN - [Grounded Language Understanding for Manipulation Instructions Using GAN-Based Classification](https://arxiv.org/abs/1801.05096) 
* LAGAN - [Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis](https://arxiv.org/abs/1701.05927) 
* LAPGAN - [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/abs/1506.05751) ([github](https://github.com/facebook/eyescream))
* LB-GAN - [Load Balanced GANs for Multi-view Face Image Synthesis](http://arxiv.org/abs/1802.07447) 
* LBT - [Learning Implicit Generative Models by Teaching Explicit Ones](https://arxiv.org/abs/1807.03870) 
* LCC-GAN - [Adversarial Learning with Local Coordinate Coding](https://arxiv.org/abs/1806.04895) 
* LD-GAN - [Linear Discriminant Generative Adversarial Networks](https://arxiv.org/abs/1707.07831) 
* LDAN - [Label Denoising Adversarial Network (LDAN) for Inverse Lighting of Face Images](https://arxiv.org/abs/1709.01993) 
* LeakGAN - [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624) 
* LeGAN - [Likelihood Estimation for Generative Adversarial Networks](https://arxiv.org/abs/1707.07530) 
* LGAN - [Global versus Localized Generative Adversarial Nets](https://arxiv.org/abs/1711.06020) 
* Lipizzaner - [Towards Distributed Coevolutionary GANs](https://arxiv.org/abs/1807.08194) 
* LR-GAN - [LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation](https://arxiv.org/abs/1703.01560v1) 
* LS-GAN - [Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities](https://arxiv.org/abs/1701.06264) 
* LSGAN - [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076v3) 
* M-AAE - [Mask-aware Photorealistic Face Attribute Manipulation](https://arxiv.org/abs/1804.08882) 
* MAD-GAN - [Multi-Agent Diverse Generative Adversarial Networks](https://arxiv.org/abs/1704.02906) 
* MAGAN - [MAGAN: Margin Adaptation for Generative Adversarial Networks](https://arxiv.org/abs/1704.03817v1) 
* MAGAN - [MAGAN: Aligning Biological Manifolds](https://arxiv.org/abs/1803.00385) 
* MalGAN - [Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN](https://arxiv.org/abs/1702.05983v1) 
* MaliGAN - [Maximum-Likelihood Augmented Discrete Generative Adversarial Networks](https://arxiv.org/abs/1702.07983) 
* manifold-WGAN - [Manifold-valued Image Generation with Wasserstein Adversarial Networks](https://arxiv.org/abs/1712.01551) 
* MARTA-GAN - [Deep Unsupervised Representation Learning for Remote Sensing Images](https://arxiv.org/abs/1612.08879) 
* MaskGAN - [MaskGAN: Better Text Generation via Filling in the ______ ](https://arxiv.org/abs/1801.07736 ) 
* MC-GAN - [Multi-Content GAN for Few-Shot Font Style Transfer](https://arxiv.org/abs/1712.00516) ([github](https://github.com/azadis/MC-GAN))
* MC-GAN - [MC-GAN: Multi-conditional Generative Adversarial Network for Image Synthesis](https://arxiv.org/abs/1805.01123) 
* McGAN - [McGan: Mean and Covariance Feature Matching GAN](https://arxiv.org/abs/1702.08398v1) 
* MD-GAN - [Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks](https://arxiv.org/abs/1709.07592) 
* MDGAN - [Mode Regularized Generative Adversarial Networks](https://arxiv.org/abs/1612.02136) 
* MedGAN - [Generating Multi-label Discrete Electronic Health Records using Generative Adversarial Networks](https://arxiv.org/abs/1703.06490v1) 
* MedGAN - [MedGAN: Medical Image Translation using GANs](https://arxiv.org/abs/1806.06397) 
* MEGAN - [MEGAN: Mixture of Experts of Generative Adversarial Networks for Multimodal Image Generation](https://arxiv.org/abs/1805.02481) 
* MelanoGAN - [MelanoGANs: High Resolution Skin Lesion Synthesis with GANs](https://arxiv.org/abs/1804.04338) 
* memoryGAN - [Memorization Precedes Generation: Learning Unsupervised GANs with Memory Networks](https://arxiv.org/abs/1803.01500) 
* MeRGAN - [Memory Replay GANs: learning to generate images from new categories without forgetting](https://arxiv.org/abs/1809.02058) 
* MGAN - [Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](https://arxiv.org/abs/1604.04382) ([github](https://github.com/chuanli11/MGANs))
* MGGAN - [Multi-Generator Generative Adversarial Nets](https://arxiv.org/abs/1708.02556) 
* MGGAN - [MGGAN: Solving Mode Collapse using Manifold Guided Training](https://arxiv.org/abs/1804.04391) 
* MIL-GAN - [Multimodal Storytelling via Generative Adversarial Imitation Learning](https://arxiv.org/abs/1712.01455) 
* MinLGAN - [Anomaly Detection via Minimum Likelihood Generative Adversarial Networks](https://arxiv.org/abs/1808.00200) 
* MIX+GAN - [Generalization and Equilibrium in Generative Adversarial Nets (GANs)](https://arxiv.org/abs/1703.00573v3) 
* MIXGAN - [MIXGAN: Learning Concepts from Different Domains for Mixture Generation](https://arxiv.org/abs/1807.01659) 
* MLGAN - [Metric Learning-based Generative Adversarial Network](https://arxiv.org/abs/1711.02792) 
* MMC-GAN - [A Multimodal Classifier Generative Adversarial Network for Carry and Place Tasks from Ambiguous Language Instructions](https://arxiv.org/abs/1806.03847) 
* MMD-GAN - [MMD GAN: Towards Deeper Understanding of Moment Matching Network](https://arxiv.org/abs/1705.08584) ([github](https://github.com/dougalsutherland/opt-mmd))
* MMGAN - [MMGAN: Manifold Matching Generative Adversarial Network for Generating Images](https://arxiv.org/abs/1707.08273) 
* MoCoGAN - [MoCoGAN: Decomposing Motion and Content for Video Generation](https://arxiv.org/abs/1707.04993) ([github](https://github.com/sergeytulyakov/mocogan))
* Modified GAN-CLS - [Generate the corresponding Image from Text Description using Modified GAN-CLS Algorithm](https://arxiv.org/abs/1806.11302) 
* ModularGAN - [Modular Generative Adversarial Networks](https://arxiv.org/abs/1804.03343) 
* MolGAN - [MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973) 
* MPM-GAN - [Message Passing Multi-Agent GANs](https://arxiv.org/abs/1612.01294) 
* MS-GAN - [Temporal Coherency based Criteria for Predicting Video Frames using Deep Multi-stage Generative Adversarial Networks](http://papers.nips.cc/paper/7014-temporal-coherency-based-criteria-for-predicting-video-frames-using-deep-multi-stage-generative-adversarial-networks) 
* MTGAN - [MTGAN: Speaker Verification through Multitasking Triplet Generative Adversarial Networks](https://arxiv.org/abs/1803.09059) 
* MuseGAN - [MuseGAN: Symbolic-domain Music Generation and Accompaniment with Multi-track Sequential Generative Adversarial Networks](https://arxiv.org/abs/1709.06298) 
* MV-BiGAN - [Multi-view Generative Adversarial Networks](https://arxiv.org/abs/1611.02019v1) 
* N2RPP - [N2RPP: An Adversarial Network to Rebuild Plantar Pressure for ACLD Patients](https://arxiv.org/abs/1805.02825) 
* NAN - [Understanding Humans in Crowded Scenes: Deep Nested Adversarial Learning and A New Benchmark for Multi-Human Parsing](https://arxiv.org/abs/1804.03287) 
* NCE-GAN - [Dihedral angle prediction using generative adversarial networks](https://arxiv.org/abs/1803.10996) 
* ND-GAN - [Novelty Detection with GAN](https://arxiv.org/abs/1802.10560) 
* NetGAN - [NetGAN: Generating Graphs via Random Walks](https://arxiv.org/abs/1803.00816) 
* OCAN - [One-Class Adversarial Nets for Fraud Detection](https://arxiv.org/abs/1803.01798) 
* OptionGAN - [OptionGAN: Learning Joint Reward-Policy Options using Generative Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1709.06683) 
* ORGAN - [Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models ](https://arxiv.org/abs/1705.10843) 
* ORGAN - [3D Reconstruction of Incomplete Archaeological Objects Using a Generative Adversary Network](https://arxiv.org/abs/1711.06363) 
* OT-GAN - [Improving GANs Using Optimal Transport](https://arxiv.org/abs/1803.05573) 
* PacGAN - [PacGAN: The power of two samples in generative adversarial networks](https://arxiv.org/abs/1712.04086) 
* PAN - [Perceptual Adversarial Networks for Image-to-Image Transformation](https://arxiv.org/abs/1706.09138) 
* PassGAN - [PassGAN: A Deep Learning Approach for Password Guessing](https://arxiv.org/abs/1709.00440) 
* PD-WGAN - [Primal-Dual Wasserstein GAN](https://arxiv.org/abs/1805.09575) 
* Perceptual GAN - [Perceptual Generative Adversarial Networks for Small Object Detection](https://arxiv.org/abs/1706.05274) 
* PGAN - [Probabilistic Generative Adversarial Networks](https://arxiv.org/abs/1708.01886) 
* PGD-GAN - [Solving Linear Inverse Problems Using GAN Priors: An Algorithm with Provable Guarantees](https://arxiv.org/abs/1802.08406) 
* PGGAN - [Patch-Based Image Inpainting with Generative Adversarial Networks](https://arxiv.org/abs/1803.07422) 
* PIONEER - [Pioneer Networks: Progressively Growing Generative Autoencoder](https://arxiv.org/abs/1807.03026) 
* Pip-GAN - [Pipeline Generative Adversarial Networks for Facial Images Generation with Multiple Attributes](https://arxiv.org/abs/1711.10742) 
* pix2pix - [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) ([github](https://github.com/phillipi/pix2pix))
* pix2pixHD - [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585) ([github](https://github.com/NVIDIA/pix2pixHD))
* PixelGAN - [PixelGAN Autoencoders](https://arxiv.org/abs/1706.00531) 
* PM-GAN - [PM-GANs: Discriminative Representation Learning for Action Recognition Using Partial-modalities](https://arxiv.org/abs/1804.06248) 
* PN-GAN - [Pose-Normalized Image Generation for Person Re-identification](https://arxiv.org/abs/1712.02225) 
* POGAN - [Perceptually Optimized Generative Adversarial Network for Single Image Dehazing](https://arxiv.org/abs/1805.01084) 
* Pose-GAN - [The Pose Knows: Video Forecasting by Generating Pose Futures](https://arxiv.org/abs/1705.00053) 
* PP-GAN - [Privacy-Protective-GAN for Face De-identification](https://arxiv.org/abs/1806.08906) 
* PPAN - [Privacy-Preserving Adversarial Networks](https://arxiv.org/abs/1712.07008) 
* PPGN - [Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space](https://arxiv.org/abs/1612.00005) 
* PrGAN - [3D Shape Induction from 2D Views of Multiple Objects](https://arxiv.org/abs/1612.05872) 
* ProGanSR - [A Fully Progressive Approach to Single-Image Super-Resolution](https://arxiv.org/abs/1804.02900) 
* Progressive GAN - [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) ([github](https://github.com/tkarras/progressive_growing_of_gans))
* PS-GAN - [Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond](https://arxiv.org/abs/1804.02047) 
* PSGAN - [Learning Texture Manifolds with the Periodic Spatial GAN](http://arxiv.org/abs/1705.06566) 
* PSGAN - [PSGAN: A Generative Adversarial Network for Remote Sensing Image Pan-Sharpening](https://arxiv.org/abs/1805.03371) 
* PS²-GAN - [High-Quality Facial Photo-Sketch Synthesis Using Multi-Adversarial Networks](https://arxiv.org/abs/1710.10182) 
* RadialGAN - [RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks ](http://arxiv.org/abs/1802.06403) 
* RaGAN - [The relativistic discriminator: a key element missing from standard GAN](https://arxiv.org/abs/1807.00734) 
* RAN - [RAN4IQA: Restorative Adversarial Nets for No-Reference Image Quality Assessment](https://arxiv.org/abs/1712.05444) ([github]())
* RankGAN - [Adversarial Ranking for Language Generation ](https://arxiv.org/abs/1705.11001) 
* RCGAN - [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633) 
* ReConNN - [Reconstruction of Simulation-Based Physical Field with Limited Samples by Reconstruction Neural Network](https://arxiv.org/abs/1805.00528) 
* Recycle-GAN - [Recycle-GAN: Unsupervised Video Retargeting](https://arxiv.org/abs/1808.05174) 
* RefineGAN - [Compressed Sensing MRI Reconstruction with Cyclic Loss in Generative Adversarial Networks](https://arxiv.org/abs/1709.00753) 
* ReGAN - [ReGAN: RE[LAX|BAR|INFORCE] based Sequence Generation using GANs](https://arxiv.org/abs/1805.02788) ([github](https://github.com/TalkToTheGAN/REGAN))
* RegCGAN - [Unpaired Multi-Domain Image Generation via Regularized Conditional GANs](https://arxiv.org/abs/1805.02456) 
* RenderGAN - [RenderGAN: Generating Realistic Labeled Data](https://arxiv.org/abs/1611.01331) 
* Resembled GAN - [Resembled Generative Adversarial Networks: Two Domains with Similar Attributes](https://arxiv.org/abs/1807.00947) 
* ResGAN - [Generative Adversarial Network based on Resnet for Conditional Image Restoration](https://arxiv.org/abs/1707.04881) 
* RNN-WGAN - [Language Generation with Recurrent Generative Adversarial Networks without Pre-training](https://arxiv.org/abs/1706.01399) ([github](https://github.com/amirbar/rnn.wgan))
* RoCGAN - [Robust Conditional Generative Adversarial Networks](https://arxiv.org/abs/1805.08657) 
* RPGAN - [Stabilizing GAN Training with Multiple Random Projections](https://arxiv.org/abs/1705.07831) ([github](https://github.com/ayanc/rpgan))
* RTT-GAN - [Recurrent Topic-Transition GAN for Visual Paragraph Generation](https://arxiv.org/abs/1703.07022v2) 
* RWGAN - [Relaxed Wasserstein with Applications to GANs](https://arxiv.org/abs/1705.07164) 
* SAD-GAN - [SAD-GAN: Synthetic Autonomous Driving using Generative Adversarial Networks](https://arxiv.org/abs/1611.08788v1) 
* SAGA - [Generative Adversarial Learning for Spectrum Sensing](https://arxiv.org/abs/1804.00709) 
* SAGAN - [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318) 
* SalGAN - [SalGAN: Visual Saliency Prediction with Generative Adversarial Networks](https://arxiv.org/abs/1701.01081) ([github](https://github.com/imatge-upc/saliency-salgan-2017))
* SAM - [Sample-Efficient Imitation Learning via Generative Adversarial Nets](https://arxiv.org/abs/1809.02064) 
* sAOG - [Deep Structured Generative Models](https://arxiv.org/abs/1807.03877) 
* SAR-GAN - [Generating High Quality Visible Images from SAR Images Using CNNs](https://arxiv.org/abs/1802.10036) 
* SBADA-GAN - [From source to target and back: symmetric bi-directional adaptive GAN](https://arxiv.org/abs/1705.08824) 
* ScarGAN - [ScarGAN: Chained Generative Adversarial Networks to Simulate Pathological Tissue on Cardiovascular MR Scans](https://arxiv.org/abs/1808.04500) 
* SCH-GAN - [SCH-GAN: Semi-supervised Cross-modal Hashing by Generative Adversarial Network ](https://arxiv.org/abs/1802.02488 ) 
* SD-GAN - [Semantically Decomposing the Latent Spaces of Generative Adversarial Networks](https://arxiv.org/abs/1705.07904) 
* Sdf-GAN - [Sdf-GAN: Semi-supervised Depth Fusion with Multi-scale Adversarial Networks](https://arxiv.org/abs/1803.06657) 
* SEGAN - [SEGAN: Speech Enhancement Generative Adversarial Network](https://arxiv.org/abs/1703.09452v1) 
* SeGAN - [SeGAN: Segmenting and Generating the Invisible](https://arxiv.org/abs/1703.10239) 
* SegAN - [SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation](https://arxiv.org/abs/1706.01805) 
* Sem-GAN - [Sem-GAN: Semantically-Consistent Image-to-Image Translation](https://arxiv.org/abs/1807.04409) 
* SeqGAN - [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473v5) ([github](https://github.com/LantaoYu/SeqGAN))
* SeUDA - [Semantic-Aware Generative Adversarial Nets for Unsupervised Domain Adaptation in Chest X-ray Segmentation](https://arxiv.org/abs/1806.00600) 
* SG-GAN - [Semantic-aware Grad-GAN for Virtual-to-Real Urban Scene Adaption](https://arxiv.org/abs/1801.01726) ([github](https://github.com/Peilun-Li/SG-GAN))
* SG-GAN - [Sparsely Grouped Multi-task Generative Adversarial Networks for Facial Attribute Manipulation](https://arxiv.org/abs/1805.07509) 
* SGAN - [Texture Synthesis with Spatial Generative Adversarial Networks](https://arxiv.org/abs/1611.08207) 
* SGAN - [Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.04357v4) ([github](https://github.com/xunhuang1995/SGAN))
* SGAN - [Steganographic Generative Adversarial Networks](https://arxiv.org/abs/1703.05502) 
* SGAN - [SGAN: An Alternative Training of Generative Adversarial Networks](https://arxiv.org/abs/1712.02330) 
* SGAN - [CT Image Enhancement Using Stacked Generative Adversarial Networks and Transfer Learning for Lesion Segmentation Improvement](https://arxiv.org/abs/1807.07144) 
* sGAN  - [Generative Adversarial Training for MRA Image Synthesis Using Multi-Contrast MRI](https://arxiv.org/abs/1804.04366) 
* SiftingGAN - [SiftingGAN: Generating and Sifting Labeled Samples to Improve the Remote Sensing Image Scene Classification Baseline in vitro](https://arxiv.org/abs/1809.04985) 
* SiGAN - [SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face Hallucination](https://arxiv.org/abs/1807.08370) 
* SimGAN - [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828) 
* SisGAN - [Semantic Image Synthesis via Adversarial Learning](https://arxiv.org/abs/1707.06873) 
* Sketcher-Refiner GAN - [Learning Myelin Content in Multiple Sclerosis from Multimodal MRI through Adversarial Training](https://arxiv.org/abs/1804.08039) 
* SketchGAN - [Adversarial Training For Sketch Retrieval](https://arxiv.org/abs/1607.02748) 
* SketchyGAN - [SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis](https://arxiv.org/abs/1801.02753) 
* Skip-Thought GAN - [Generating Text through Adversarial Training using Skip-Thought Vectors](https://arxiv.org/abs/1808.08703) 
* SL-GAN - [Semi-Latent GAN: Learning to generate and modify facial images from attributes](https://arxiv.org/abs/1704.02166) 
* SLSR - [Sparse Label Smoothing for Semi-supervised Person Re-Identification](https://arxiv.org/abs/1809.04976) 
* SN-DCGAN - [Generative Adversarial Networks for Unsupervised Object Co-localization](https://arxiv.org/abs/1806.00236) 
* SN-GAN - [Spectral Normalization for Generative Adversarial Networks](https://drive.google.com/file/d/0B8HZ50DPgR3eSVV6YlF3XzQxSjQ/view) ([github](https://github.com/pfnet-research/chainer-gan-lib))
* SN-PatchGAN - [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589) 
* Sobolev GAN - [Sobolev GAN](https://arxiv.org/abs/1711.04894) 
* Social GAN - [Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks](https://arxiv.org/abs/1803.10892) 
* Softmax GAN - [Softmax GAN](https://arxiv.org/abs/1704.06191) 
* SoPhie - [SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints](https://arxiv.org/abs/1806.01482) 
* speech-driven animation GAN - [End-to-End Speech-Driven Facial Animation with Temporal GANs](https://arxiv.org/abs/1805.09313) 
* Spike-GAN - [Synthesizing realistic neural population activity patterns using Generative Adversarial Networks](https://arxiv.org/abs/1803.00338) 
* Splitting GAN - [Class-Splitting Generative Adversarial Networks](https://arxiv.org/abs/1709.07359) 
* SR-CNN-VAE-GAN - [Semi-Recurrent CNN-based VAE-GAN for Sequential Data Generation](https://arxiv.org/abs/1806.00509) ([github](https://github.com/makbari7/SR-CNN-VAE-GAN))
* SRGAN - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) 
* SRPGAN - [SRPGAN: Perceptual Generative Adversarial Network for Single Image Super Resolution](https://arxiv.org/abs/1712.05927) 
* SS-GAN - [Semi-supervised Conditional GANs](https://arxiv.org/abs/1708.05789) 
* ss-InfoGAN - [Guiding InfoGAN with Semi-Supervision](https://arxiv.org/abs/1707.04487) 
* SSGAN - [SSGAN: Secure Steganography Based on Generative Adversarial Networks](https://arxiv.org/abs/1707.01613) 
* SSL-GAN - [Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks](https://arxiv.org/abs/1611.06430v1) 
* ST-CGAN - [Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal](https://arxiv.org/abs/1712.02478) 
* ST-GAN - [Style Transfer Generative Adversarial Networks: Learning to Play Chess Differently](https://arxiv.org/abs/1702.06762) 
* ST-GAN - [ST-GAN: Spatial Transformer Generative Adversarial Networks for Image Compositing](https://arxiv.org/abs/1803.01837) 
* StackGAN - [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.03242v1) ([github](https://github.com/hanzhanggit/StackGAN))
* StainGAN - [StainGAN: Stain Style Transfer for Digital Histological Images](https://arxiv.org/abs/1804.01601) 
* StarGAN - [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) ([github](https://github.com/yunjey/StarGAN))
* StarGAN-VC - [StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks ](https://arxiv.org/abs/1806.02169) 
* SteinGAN - [Learning Deep Energy Models: Contrastive Divergence vs. Amortized MLE](https://arxiv.org/abs/1707.00797) 
* StepGAN - [Improving Conditional Sequence Generative Adversarial Networks by Stepwise Evaluation](https://arxiv.org/abs/1808.05599) 
* Super-FAN - [Super-FAN: Integrated facial landmark localization and super-resolution of real-world low resolution faces in arbitrary poses with GANs](https://arxiv.org/abs/1712.02765) 
* SVSGAN - [SVSGAN: Singing Voice Separation via Generative Adversarial Network](https://arxiv.org/abs/1710.11428) 
* SWGAN - [Solving Approximate Wasserstein GANs to Stationarity](https://arxiv.org/abs/1802.08249) 
* SyncGAN - [SyncGAN: Synchronize the Latent Space of Cross-modal Generative Adversarial Networks](https://arxiv.org/abs/1804.00410) 
* S^2GAN - [Generative Image Modeling using Style and Structure Adversarial Networks](https://arxiv.org/abs/1603.05631v2) 
* T2Net - [T2Net: Synthetic-to-Realistic Translation for Solving Single-Image Depth Estimation Tasks](https://arxiv.org/abs/1808.01454) 
* table-GAN - [Data Synthesis based on Generative Adversarial Networks](https://arxiv.org/abs/1806.03384) 
* TAC-GAN - [TAC-GAN - Text Conditioned Auxiliary Classifier Generative Adversarial Network](https://arxiv.org/abs/1703.06412v2) ([github](https://github.com/dashayushman/TAC-GAN))
* TAN - [Outline Colorization through Tandem Adversarial Networks](https://arxiv.org/abs/1704.08834) 
* tcGAN - [Cross-modal Hallucination for Few-shot Fine-grained Recognition](https://arxiv.org/abs/1806.05147) 
* TD-GAN - [Task Driven Generative Modeling for Unsupervised Domain Adaptation: Application to X-ray Image Segmentation](https://arxiv.org/abs/1806.07201) 
* tempCycleGAN - [Improving Surgical Training Phantoms by Hyperrealism: Deep Unpaired Image-to-Image Translation from Real Surgeries](https://arxiv.org/abs/1806.03627) 
* tempoGAN - [tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow](https://arxiv.org/abs/1801.09710) 
* TequilaGAN - [TequilaGAN: How to easily identify GAN samples](https://arxiv.org/abs/1807.04919) 
* Text2Shape - [Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings](https://arxiv.org/abs/1803.08495) 
* textGAN - [Generating Text via Adversarial Training](https://zhegan27.github.io/Papers/textGAN_nips2016_workshop.pdf) 
* TextureGAN - [TextureGAN: Controlling Deep Image Synthesis with Texture Patches](https://arxiv.org/abs/1706.02823) 
* TGAN - [Temporal Generative Adversarial Nets](https://arxiv.org/abs/1611.06624v1) 
* TGAN - [Tensorizing Generative Adversarial Nets](https://arxiv.org/abs/1710.10772) 
* TGAN - [Tensor-Generative Adversarial Network with Two-dimensional Sparse Coding: Application to Real-time Indoor Localization](https://arxiv.org/abs/1711.02666) 
* TGANs-C - [To Create What You Tell: Generating Videos from Captions](https://arxiv.org/abs/1804.08264) 
* tiny-GAN - [Analysis of Nonautonomous Adversarial Systems](https://arxiv.org/abs/1803.05045) 
* TP-GAN - [Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](https://arxiv.org/abs/1704.04086) 
* TreeGAN - [TreeGAN: Syntax-Aware Sequence Generation with Generative Adversarial Networks](https://arxiv.org/abs/1808.07582) 
* Triple-GAN - [Triple Generative Adversarial Nets](https://arxiv.org/abs/1703.02291v2) 
* tripletGAN - [TripletGAN: Training Generative Model with Triplet Loss](https://arxiv.org/abs/1711.05084) 
* TV-GAN - [TV-GAN: Generative Adversarial Network Based Thermal to Visible Face Recognition](https://arxiv.org/abs/1712.02514) 
* Twin-GAN - [Twin-GAN -- Unpaired Cross-Domain Image Translation with Weight-Sharing GANs](https://arxiv.org/abs/1809.00946) 
* UGACH - [Unsupervised Generative Adversarial Cross-modal Hashing](https://arxiv.org/abs/1712.00358) 
* UGAN - [Enhancing Underwater Imagery using Generative Adversarial Networks](https://arxiv.org/abs/1801.04011) 
* Unim2im - [Unsupervised Image-to-Image Translation with Generative Adversarial Networks ](https://arxiv.org/abs/1701.02676) ([github](http://github.com/zsdonghao/Unsup-Im2Im))
* UNIT - [Unsupervised Image-to-image Translation Networks](https://arxiv.org/abs/1703.00848) ([github](https://github.com/mingyuliutw/UNIT))
* Unrolled GAN - [Unrolled Generative Adversarial Networks](https://arxiv.org/abs/1611.02163) ([github](https://github.com/poolio/unrolled_gan))
* UT-SCA-GAN - [Spatial Image Steganography Based on Generative Adversarial Network](https://arxiv.org/abs/1804.07939) 
* UV-GAN - [UV-GAN: Adversarial Facial UV Map Completion for Pose-invariant Face Recognition](https://arxiv.org/abs/1712.04695) 
* VA-GAN - [Visual Feature Attribution using Wasserstein GANs](https://arxiv.org/abs/1711.08998) 
* VAC+GAN  - [Versatile Auxiliary Classifier with Generative Adversarial Network (VAC+GAN), Multi Class Scenarios](https://arxiv.org/abs/1806.07751) 
* VAE-GAN - [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300) 
* VariGAN - [Multi-View Image Generation from a Single-View](https://arxiv.org/abs/1704.04886) 
* VAW-GAN - [Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/1704.00849) 
* VEEGAN - [VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning](https://arxiv.org/abs/1705.07761) ([github](https://github.com/akashgit/VEEGAN))
* VGAN - [Generating Videos with Scene Dynamics](https://arxiv.org/abs/1609.02612) ([github](https://github.com/cvondrick/videogan))
* VGAN - [Generative Adversarial Networks as Variational Training of Energy Based Models](https://arxiv.org/abs/1611.01799) ([github](https://github.com/Shuangfei/vgan))
* VGAN - [Text Generation Based on Generative Adversarial Nets with Latent Variable](https://arxiv.org/abs/1712.00170) 
* ViGAN - [Image Generation and Editing with Variational Info Generative Adversarial Networks](https://arxiv.org/abs/1701.04568v1) 
* VIGAN - [VIGAN: Missing View Imputation with Generative Adversarial Networks](https://arxiv.org/abs/1708.06724) 
* VoiceGAN - [Voice Impersonation using Generative Adversarial Networks ](http://arxiv.org/abs/1802.06840) 
* VOS-GAN - [VOS-GAN: Adversarial Learning of Visual-Temporal Dynamics for Unsupervised Dense Prediction in Videos](https://arxiv.org/abs/1803.09092) 
* VRAL - [Variance Regularizing Adversarial Learning](https://arxiv.org/abs/1707.00309) 
* WaterGAN - [WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images](https://arxiv.org/abs/1702.07392v1) 
* WaveGAN - [Synthesizing Audio with Generative Adversarial Networks ](https://arxiv.org/abs/1802.04208) 
* WaveletGLCA-GAN - [Global and Local Consistent Wavelet-domain Age Synthesis](https://arxiv.org/abs/1809.07764) 
* weGAN - [Generative Adversarial Nets for Multiple Text Corpora](https://arxiv.org/abs/1712.09127) 
* WGAN - [Wasserstein GAN](https://arxiv.org/abs/1701.07875v2) ([github](https://github.com/martinarjovsky/WassersteinGAN))
* WGAN-CLS - [Text to Image Synthesis Using Generative Adversarial Networks](https://arxiv.org/abs/1805.00676) 
* WGAN-GP - [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) ([github](https://github.com/igul222/improved_wgan_training))
* WGAN-L1 - [Subsampled Turbulence Removal Network](https://arxiv.org/abs/1807.04418) 
* WS-GAN - [Weakly Supervised Generative Adversarial Networks for 3D Reconstruction ](https://arxiv.org/abs/1705.10904) 
* X-GANs - [X-GANs: Image Reconstruction Made Easy for Extreme Cases](https://arxiv.org/abs/1808.04432) 
* XGAN - [XGAN: Unsupervised Image-to-Image Translation for many-to-many Mappings](https://arxiv.org/abs/1711.05139) 
* ZipNet-GAN - [ZipNet-GAN: Inferring Fine-grained Mobile Traffic Patterns via a Generative Adversarial Neural Network](https://arxiv.org/abs/1711.02413) 
* α-GAN - [Variational Approaches for Auto-Encoding Generative Adversarial Networks](https://arxiv.org/abs/1706.04987) ([github](https://github.com/victor-shepardson/alpha-GAN))
* β-GAN - [Annealed Generative Adversarial Networks](https://arxiv.org/abs/1705.07505) 
* Δ-GAN - [Triangle Generative Adversarial Networks](https://arxiv.org/abs/1709.06548) 



----------

### :electric_plug: Frameworks & Libraries (Descending order based on GitHub stars)
- Tensorflow by Google  [C++ and CUDA]: [[homepage]](https://www.tensorflow.org/) [[github]](https://github.com/tensorflow/tensorflow)
- Caffe by Berkeley Vision and Learning Center (BVLC)  [C++]: [[homepage]](http://caffe.berkeleyvision.org/) [[github]](https://github.com/BVLC/caffe) [[Installation Instructions]](Caffe_Installation/README.md)
- Keras by François Chollet  [Python]: [[homepage]](https://keras.io/) [[github]](https://github.com/fchollet/keras)
- Microsoft Cognitive Toolkit - CNTK  [C++]: [[homepage]](https://www.microsoft.com/en-us/research/product/cognitive-toolkit/) [[github]](https://github.com/Microsoft/CNTK)
- MXNet adapted by Amazon  [C++]: [[homepage]](http://mxnet.io/) [[github]](https://github.com/dmlc/mxnet)
- Torch by Collobert, Kavukcuoglu & Clement Farabet, widely used by Facebook  [Lua]: [[homepage]](http://torch.ch/) [[github]](https://github.com/torch) 	
- Convnetjs by Andrej Karpathy [JavaScript]: [[homepage]](http://cs.stanford.edu/people/karpathy/convnetjs/) [[github]](https://github.com/karpathy/convnetjs)
- Theano by Université de Montréal  [Python]: [[homepage]](http://deeplearning.net/software/theano/) [[github]](https://github.com/Theano/Theano) 	
- Deeplearning4j by startup Skymind  [Java]: [[homepage]](https://deeplearning4j.org/) [[github]](https://github.com/deeplearning4j/deeplearning4j) 	
- Caffe2 by Facebook Open Source [C++ & Python]: [[github]](https://github.com/caffe2/caffe2) [[web]](https://caffe2.ai/)
- Paddle by Baidu  [C++]: [[homepage]](http://www.paddlepaddle.org/) [[github]](https://github.com/PaddlePaddle/Paddle)
- Deep Scalable Sparse Tensor Network Engine (DSSTNE) by Amazon  [C++]: [[github]](https://github.com/amznlabs/amazon-dsstne)
- Neon by Nervana Systems  [Python & Sass]: [[homepage]](http://neon.nervanasys.com/docs/latest/) [[github]](https://github.com/NervanaSystems/neon) 	
- Chainer  [Python]: [[homepage]](http://chainer.org/) [[github]](https://github.com/pfnet/chainer) 	
- h2o  [Java]: [[homepage]](http://www.h2o.ai/) [[github]](https://github.com/h2oai/h2o-3) 	
- Brainstorm by Istituto Dalle Molle di Studi sull’Intelligenza Artificiale (IDSIA)  [Python]: [[github]](https://github.com/IDSIA/brainstorm)
- Matconvnet by Andrea Vedaldi  [Matlab]: [[homepage]](http://www.vlfeat.org/matconvnet/) [[github]](https://github.com/vlfeat/matconvnet) 	
----




[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [@thomasfuchs]: <http://twitter.com/thomasfuchs>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [keymaster.js]: <https://github.com/madrobby/keymaster>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]:  <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   
   Liquid Warping GAN with Attention: A Unified Framework for Human Image Synthesis,

**Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Surfaces**  
[Towaki Takikawa*](https://tovacinni.github.io), [Joey Litalien*](https://joeylitalien.github.io), [Kangxue Xin](https://kangxue.org/), [Karsten Kreis](https://scholar.google.de/citations?user=rFd-DiAAAAAJ), [Charles Loop](https://research.nvidia.com/person/charles-loop), [Derek Nowrouzezahrai](http://www.cim.mcgill.ca/~derek/), [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/), [Morgan McGuire](https://casual-effects.com/), and [Sanja Fidler](https://www.cs.toronto.edu/~fidler/)In submission, 2021

**[[Paper](https://arxiv.org/abs/2101.10994)] [[Bibtex](https://nv-tlabs.github.io/nglod/assets/nglod.bib)] [[Project Page](https://nv-tlabs.github.io/nglod/)]**


** best paper of GAN 

|Year	|Month	|Abbr.	|Title|	Arxiv	|Official_Code	
:---:|:---:|:---:|:---:|:---:|:---:
2014	6	GAN	Generative Adversarial Networks	https://arxiv.org/abs/1406.2661	https://github.com/goodfeli/adversarial
2014	11 CGAN	Conditional Generative Adversarial Nets	https://arxiv.org/abs/1411.1784
2015	6	LAPGAN	Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks	https://arxiv.org/abs/1506.05751	https://github.com/facebook/eyescream	
2015	11 CatGAN	Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks	https://arxiv.org/abs/1511.06390v2	
2015	11 DCGAN	Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks	https://arxiv.org/abs/1511.06434	https://github.com/Newmu/dcgan_code	
2015	12	VAE-GAN	Autoencoding beyond pixels using a learned similarity metric	https://arxiv.org/abs/1512.09300	-			
2016	2	GRAN	Generating images with recurrent adversarial networks	https://arxiv.org/abs/1602.05110	https://github.com/jiwoongim/GRAN
2016	3	S^2GAN	Generative Image Modeling using Style and Structure Adversarial Networks	https://arxiv.org/abs/1603.05631v2	-			
2016	4	MGAN	Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks	https://arxiv.org/abs/1604.04382	https://github.com/chuanli11/MGANs
2016	5	BiGAN	Adversarial Feature Learning	https://arxiv.org/abs/1605.09782v7	-			
2016	5	GAN-CLS	Generative Adversarial Text to Image Synthesis	https://arxiv.org/abs/1605.05396	https://github.com/reedscot/icml2016	39	677	155
2016	6	ALI	Adversarially Learned Inference	https://arxiv.org/abs/1606.00704	https://github.com/IshmaelBelghazi/ALI	16	262	72
2016	6	CoGAN	Coupled Generative Adversarial Networks	https://arxiv.org/abs/1606.07536v2	-			
2016	6	f-GAN	f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization	https://arxiv.org/abs/1606.00709	-			
2016	6	Improved GAN	Improved Techniques for Training GANs	https://arxiv.org/abs/1606.03498	https://github.com/openai/improved-gan	143	1462	445
2016	6	InfoGAN	InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets	https://arxiv.org/abs/1606.03657v1	https://github.com/openai/InfoGAN	100	666	211
2016	7	SketchGAN	Adversarial Training For Sketch Retrieval	https://arxiv.org/abs/1607.02748	-			
2016	9	Context-RNN-GAN	Contextual RNN-GANs for Abstract Reasoning Diagram Generation	https://arxiv.org/abs/1609.09444	-	
2016	9	EBGAN	Energy-based Generative Adversarial Network	https://arxiv.org/abs/1609.03126v4	-			
2016	9	IAN	Neural Photo Editing with Introspective Adversarial Networks	https://arxiv.org/abs/1609.07093	https://github.com/ajbrock/Neural-Photo-Editor	74	1753	152
2016	9	iGAN	Generative Visual Manipulation on the Natural Image Manifold	https://arxiv.org/abs/1609.03552v2	https://github.com/junyanz/iGAN	151	2984	425
2016	9	SeqGAN	SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient	https://arxiv.org/abs/1609.05473v5	https://github.com/LantaoYu/SeqGAN	71	1243	464
2016	9	SRGAN	Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network	https://arxiv.org/abs/1609.04802	-			
2016	9	VGAN	Generating Videos with Scene Dynamics	https://arxiv.org/abs/1609.02612	https://github.com/cvondrick/videogan	34	556	114
2016	10	3D-GAN	Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling	https://arxiv.org/abs/1610.07584	https://github.com/zck119/3dgan-release	44	437	122
2016	10	AC-GAN	Conditional Image Synthesis With Auxiliary Classifier GANs	https://arxiv.org/abs/1610.09585	-			
2016	10	AffGAN	Amortised MAP Inference for Image Super-resolution	https://arxiv.org/abs/1610.04490	-			
2016	10	GAWWN	Learning What and Where to Draw	https://arxiv.org/abs/1610.02454	https://github.com/reedscot/nips2016	22	299	55
2016	11	b-GAN	Generative Adversarial Nets from a Density Ratio Estimation Perspective	https://arxiv.org/abs/1610.02920	-			
2016	11	C-RNN-GAN	C-RNN-GAN: Continuous recurrent neural networks with adversarial training	https://arxiv.org/abs/1611.09904	https://github.com/olofmogren/c-rnn-gan/	13	188	67
2016	11	CC-GAN	Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks	https://arxiv.org/abs/1611.06430	https://github.com/edenton/cc-gan	5	16	2
2016	11	DTN	Unsupervised Cross-Domain Image Generation	https://arxiv.org/abs/1611.02200	-			
2016	11	GMAN	Generative Multi-Adversarial Networks	http://arxiv.org/abs/1611.01673	-			
2016	11	IcGAN	Invertible Conditional GANs for image editing	https://arxiv.org/abs/1611.06355	https://github.com/Guim3/IcGAN	6	188	285
2016	11	LSGAN	Least Squares Generative Adversarial Networks	https://arxiv.org/abs/1611.04076v3	-			
2016	11	MV-BiGAN	Multi-view Generative Adversarial Networks	https://arxiv.org/abs/1611.02019v1	-			
2016	11	pix2pix	Image-to-Image Translation with Conditional Adversarial Networks	https://arxiv.org/abs/1611.07004	https://github.com/phillipi/pix2pix	248	5260	839
2016	11	RenderGAN	RenderGAN: Generating Realistic Labeled Data	https://arxiv.org/abs/1611.01331	-			
2016	11	SAD-GAN	SAD-GAN: Synthetic Autonomous Driving using Generative Adversarial Networks	https://arxiv.org/abs/1611.08788v1	-			
2016	11	SGAN	Texture Synthesis with Spatial Generative Adversarial Networks	https://arxiv.org/abs/1611.08207	-			
2016	11	SSL-GAN	Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks	https://arxiv.org/abs/1611.06430v1	-			
2016	11	TGAN	Temporal Generative Adversarial Nets	https://arxiv.org/abs/1611.06624v1	-			
2016	11	Unrolled GAN	Unrolled Generative Adversarial Networks	https://arxiv.org/abs/1611.02163	https://github.com/poolio/unrolled_gan	14	234	40
2016	11	VGAN	Generative Adversarial Networks as Variational Training of Energy Based Models	https://arxiv.org/abs/1611.01799	https://github.com/Shuangfei/vgan	3	15	10
2016	12	AL-CGAN	Learning to Generate Images of Outdoor Scenes from Attributes and Semantic Layouts	https://arxiv.org/abs/1612.00215	-			
2016	12	MARTA-GAN	Deep Unsupervised Representation Learning for Remote Sensing Images	https://arxiv.org/abs/1612.08879	-			
2016	12	MDGAN	Mode Regularized Generative Adversarial Networks	https://arxiv.org/abs/1612.02136	-			
2016	12	MPM-GAN	Message Passing Multi-Agent GANs	https://arxiv.org/abs/1612.01294	-			
2016	12	PPGN	Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space	https://arxiv.org/abs/1612.00005	-			
2016	12	PrGAN	3D Shape Induction from 2D Views of Multiple Objects	https://arxiv.org/abs/1612.05872	-			
2016	12	SGAN	Stacked Generative Adversarial Networks	https://arxiv.org/abs/1612.04357v4	https://github.com/xunhuang1995/SGAN	8	202	49
2016	12	SimGAN	Learning from Simulated and Unsupervised Images through Adversarial Training	https://arxiv.org/abs/1612.07828	-			
2016	12	StackGAN	StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks	https://arxiv.org/abs/1612.03242v1	https://github.com/hanzhanggit/StackGAN	59	1201	295
2016	12	textGAN	Generating Text via Adversarial Training	https://zhegan27.github.io/Papers/textGAN_nips2016_workshop.pdf	-			
2017	1	AdaGAN	AdaGAN: Boosting Generative Models	https://arxiv.org/abs/1701.02386v1	-			
2017	1	ID-CGAN	Image De-raining Using a Conditional Generative Adversarial Network	https://arxiv.org/abs/1701.05957v3	-			
2017	1	LAGAN	Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis	https://arxiv.org/abs/1701.05927	-			
2017	1	LS-GAN	Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities	https://arxiv.org/abs/1701.06264	-			
2017	1	SalGAN	SalGAN: Visual Saliency Prediction with Generative Adversarial Networks	https://arxiv.org/abs/1701.01081	https://github.com/imatge-upc/saliency-salgan-2017	15	180	80
2017	1	Unim2im	Unsupervised Image-to-Image Translation with Generative Adversarial Networks 	https://arxiv.org/abs/1701.02676	http://github.com/zsdonghao/Unsup-Im2Im	7	48	12
2017	1	ViGAN	Image Generation and Editing with Variational Info Generative Adversarial Networks	https://arxiv.org/abs/1701.04568v1	-			
2017	1	WGAN	Wasserstein GAN	https://arxiv.org/abs/1701.07875v2	https://github.com/martinarjovsky/WassersteinGAN	99	1939	469
2017	2	acGAN	Face Aging With Conditional Generative Adversarial Networks	https://arxiv.org/abs/1702.01983	-			
2017	2	ArtGAN	ArtGAN: Artwork Synthesis with Conditional Categorial GANs	https://arxiv.org/abs/1702.03410	-			
2017	2	Bayesian GAN	Deep and Hierarchical Implicit Models	https://arxiv.org/abs/1702.08896	-			
2017	2	BS-GAN	Boundary-Seeking Generative Adversarial Networks	https://arxiv.org/abs/1702.08431v1	-			
2017	2	MalGAN	Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN	https://arxiv.org/abs/1702.05983v1	-			
2017	2	MaliGAN	Maximum-Likelihood Augmented Discrete Generative Adversarial Networks	https://arxiv.org/abs/1702.07983	-			
2017	2	McGAN	McGan: Mean and Covariance Feature Matching GAN	https://arxiv.org/abs/1702.08398v1	-			
2017	2	ST-GAN	Style Transfer Generative Adversarial Networks: Learning to Play Chess Differently	https://arxiv.org/abs/1702.06762	-			
2017	2	WaterGAN	WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images	https://arxiv.org/abs/1702.07392v1	-			
2017	3	AEGAN	Learning Inverse Mapping by Autoencoder based Generative Adversarial Nets	https://arxiv.org/abs/1703.10094	-			
2017	3	AM-GAN	Activation Maximization Generative Adversarial Nets	https://arxiv.org/abs/1703.02000	-			
2017	3	AnoGAN	Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery	https://arxiv.org/abs/1703.05921v1	-			
2017	3	BEGAN	BEGAN: Boundary Equilibrium Generative Adversarial Networks	https://arxiv.org/abs/1703.10717	-			
2017	3	CS-GAN	Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets	https://arxiv.org/abs/1703.04887	-			
2017	3	CVAE-GAN	CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training	https://arxiv.org/abs/1703.10155	-			
2017	3	CycleGAN	Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks	https://arxiv.org/abs/1703.10593	https://github.com/junyanz/CycleGAN	345	7114	1099
2017	3	DiscoGAN	Learning to Discover Cross-Domain Relations with Generative Adversarial Networks	https://arxiv.org/abs/1703.05192v1	-			
2017	3	GP-GAN	GP-GAN: Towards Realistic High-Resolution Image Blending	https://arxiv.org/abs/1703.07195	https://github.com/wuhuikai/GP-GAN	6	127	29
2017	3	LR-GAN	LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation	https://arxiv.org/abs/1703.01560v1	-			
2017	3	MedGAN	Generating Multi-label Discrete Electronic Health Records using Generative Adversarial Networks	https://arxiv.org/abs/1703.06490v1	-			
2017	3	MIX+GAN	Generalization and Equilibrium in Generative Adversarial Nets (GANs)	https://arxiv.org/abs/1703.00573v3	-			
2017	3	RTT-GAN	Recurrent Topic-Transition GAN for Visual Paragraph Generation	https://arxiv.org/abs/1703.07022v2	-			
2017	3	SEGAN	SEGAN: Speech Enhancement Generative Adversarial Network	https://arxiv.org/abs/1703.09452v1	-			
2017	3	SeGAN	SeGAN: Segmenting and Generating the Invisible	https://arxiv.org/abs/1703.10239	-			
2017	3	SGAN	Steganographic Generative Adversarial Networks	https://arxiv.org/abs/1703.05502	-			
2017	3	TAC-GAN	TAC-GAN - Text Conditioned Auxiliary Classifier Generative Adversarial Network	https://arxiv.org/abs/1703.06412v2	https://github.com/dashayushman/TAC-GAN	3	36	15
2017	3	Triple-GAN	Triple Generative Adversarial Nets	https://arxiv.org/abs/1703.02291v2	-			
2017	3	UNIT	Unsupervised Image-to-image Translation Networks	https://arxiv.org/abs/1703.00848	https://github.com/mingyuliutw/UNIT	55	1094	201
2017	4	DualGAN	DualGAN: Unsupervised Dual Learning for Image-to-Image Translation	https://arxiv.org/abs/1704.02510v1	-			
2017	4	FF-GAN	Towards Large-Pose Face Frontalization in the Wild	https://arxiv.org/abs/1704.06244	-			
2017	4	GoGAN	Gang of GANs: Generative Adversarial Networks with Maximum Margin Ranking	https://arxiv.org/abs/1704.04865	-			
2017	4	MAD-GAN	Multi-Agent Diverse Generative Adversarial Networks	https://arxiv.org/abs/1704.02906	-			
2017	4	MAGAN	MAGAN: Margin Adaptation for Generative Adversarial Networks	https://arxiv.org/abs/1704.03817v1	-			
2017	4	SL-GAN	Semi-Latent GAN: Learning to generate and modify facial images from attributes	https://arxiv.org/abs/1704.02166	-			
2017	4	Softmax GAN	Softmax GAN	https://arxiv.org/abs/1704.06191	-			
2017	4	TAN	Outline Colorization through Tandem Adversarial Networks	https://arxiv.org/abs/1704.08834	-			
2017	4	TP-GAN	Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis	https://arxiv.org/abs/1704.04086	-			
2017	4	VariGAN	Multi-View Image Generation from a Single-View	https://arxiv.org/abs/1704.04886	-			
2017	4	VAW-GAN	Voice Conversion from Unaligned Corpora using Variational Autoencoding Wasserstein Generative Adversarial Networks	https://arxiv.org/abs/1704.00849	-			
2017	4	WGAN-GP	Improved Training of Wasserstein GANs	https://arxiv.org/abs/1704.00028	https://github.com/igul222/improved_wgan_training	67	1277	415
2017	4	β-GAN	Annealed Generative Adversarial Networks	https://arxiv.org/abs/1705.07505	-			
2017	5	Bayesian GAN	Bayesian GAN	https://arxiv.org/abs/1705.09558	https://github.com/andrewgordonwilson/bayesgan/	53	929	148
2017	5	CaloGAN	CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks	https://arxiv.org/abs/1705.02355	https://github.com/hep-lbdl/CaloGAN	9	64	26
2017	5	Conditional cycleGAN	Conditional CycleGAN for Attribute Guided Face Image Generation	https://arxiv.org/abs/1705.09966	-			
2017	5	Cramèr GAN 	The Cramer Distance as a Solution to Biased Wasserstein Gradients	https://arxiv.org/abs/1705.10743	-			
2017	5	DR-GAN	Representation Learning by Rotating Your Faces	https://arxiv.org/abs/1705.11136	-			
2017	5	DRAGAN	How to Train Your DRAGAN	https://arxiv.org/abs/1705.07215	https://github.com/kodalinaveen3/DRAGAN	11	143	17
2017	5	ED//GAN	Stabilizing Training of Generative Adversarial Networks through Regularization	https://arxiv.org/abs/1705.09367	-			
2017	5	EGAN	Enhanced Experience Replay Generation for Efficient Reinforcement Learning	https://arxiv.org/abs/1705.08245	-			
2017	5	Fisher GAN	Fisher GAN	https://arxiv.org/abs/1705.09675	-			
2017	5	Flow-GAN	Flow-GAN: Bridging implicit and prescribed learning in generative models	https://arxiv.org/abs/1705.08868	-			
2017	5	GeneGAN	GeneGAN: Learning Object Transfiguration and Attribute Subspace from Unpaired Data	https://arxiv.org/abs/1705.04932	https://github.com/Prinsphield/GeneGAN	9	106	22
2017	5	Geometric GAN	Geometric GAN	https://arxiv.org/abs/1705.02894	-			
2017	5	IRGAN	IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval models	https://arxiv.org/abs/1705.10513v1	-			
2017	5	MMD-GAN	MMD GAN: Towards Deeper Understanding of Moment Matching Network	https://arxiv.org/abs/1705.08584	https://github.com/dougalsutherland/opt-mmd	7	110	34
2017	5	ORGAN	Objective-Reinforced Generative Adversarial Networks (ORGAN) for Sequence Generation Models 	https://arxiv.org/abs/1705.10843	-			
2017	5	Pose-GAN	The Pose Knows: Video Forecasting by Generating Pose Futures	https://arxiv.org/abs/1705.00053	-			
2017	5	PSGAN	Learning Texture Manifolds with the Periodic Spatial GAN	http://arxiv.org/abs/1705.06566	-			
2017	5	RankGAN	Adversarial Ranking for Language Generation 	https://arxiv.org/abs/1705.11001	-			
2017	5	RPGAN	Stabilizing GAN Training with Multiple Random Projections	https://arxiv.org/abs/1705.07831	https://github.com/ayanc/rpgan	2	15	4
2017	5	RWGAN	Relaxed Wasserstein with Applications to GANs	https://arxiv.org/abs/1705.07164	-			
2017	5	SBADA-GAN	From source to target and back: symmetric bi-directional adaptive GAN	https://arxiv.org/abs/1705.08824	-			
2017	5	SD-GAN	Semantically Decomposing the Latent Spaces of Generative Adversarial Networks	https://arxiv.org/abs/1705.07904	-			
2017	5	VEEGAN	VEEGAN: Reducing Mode Collapse in GANs using Implicit Variational Learning	https://arxiv.org/abs/1705.07761	https://github.com/akashgit/VEEGAN	2	11	6
2017	5	WS-GAN	Weakly Supervised Generative Adversarial Networks for 3D Reconstruction 	https://arxiv.org/abs/1705.10904	-			
2017	6	ARAE	Adversarially Regularized Autoencoders for Generating Discrete Structures	https://arxiv.org/abs/1706.04223	https://github.com/jakezhaojb/ARAE	17	260	62
2017	6	BCGAN	Bayesian Conditional Generative Adverserial Networks	https://arxiv.org/abs/1706.05477	-			
2017	6	CAN	"CAN: Creative Adversarial Networks, Generating Art by Learning About Styles and Deviating from Style Norms"	https://arxiv.org/abs/1706.07068	-			
2017	6	Chekhov GAN	An Online Learning Approach to Generative Adversarial Networks	https://arxiv.org/abs/1706.03269	-			
2017	6	crVAE-GAN	Channel-Recurrent Variational Autoencoders	https://arxiv.org/abs/1706.03729	-			
2017	6	DeliGAN	DeLiGAN : Generative Adversarial Networks for Diverse and Limited Data	https://arxiv.org/abs/1706.02071	https://github.com/val-iisc/deligan	4	74	25
2017	6	DistanceGAN	One-Sided Unsupervised Domain Mapping	https://arxiv.org/abs/1706.00826	-			
2017	6	DSP-GAN	Depth Structure Preserving Scene Image Generation	https://arxiv.org/abs/1706.00212	-			
2017	6	Dualing GAN	Dualing GANs	https://arxiv.org/abs/1706.06216	-			
2017	6	Fila-GAN	Synthesizing Filamentary Structured Images with GANs	https://arxiv.org/abs/1706.02185	-			
2017	6	GANCS	Deep Generative Adversarial Networks for Compressed Sensing Automates MRI	https://arxiv.org/abs/1706.00051	-			
2017	6	GMM-GAN	Towards Understanding the Dynamics of Generative Adversarial Networks	https://arxiv.org/abs/1706.09884	-			
2017	6	IWGAN	On Unifying Deep Generative Models	https://arxiv.org/abs/1706.00550	-			
2017	6	PAN	Perceptual Adversarial Networks for Image-to-Image Transformation	https://arxiv.org/abs/1706.09138	-			
2017	6	Perceptual GAN	Perceptual Generative Adversarial Networks for Small Object Detection	https://arxiv.org/abs/1706.05274	-			
2017	6	PixelGAN	PixelGAN Autoencoders	https://arxiv.org/abs/1706.00531	-			
2017	6	RCGAN	Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs	https://arxiv.org/abs/1706.02633	-			
2017	6	RNN-WGAN	Language Generation with Recurrent Generative Adversarial Networks without Pre-training	https://arxiv.org/abs/1706.01399	https://github.com/amirbar/rnn.wgan	19	205	63
2017	6	SegAN	SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation	https://arxiv.org/abs/1706.01805	-			
2017	6	TextureGAN	TextureGAN: Controlling Deep Image Synthesis with Texture Patches	https://arxiv.org/abs/1706.02823	-			
2017	6	α-GAN	Variational Approaches for Auto-Encoding Generative Adversarial Networks	https://arxiv.org/abs/1706.04987	https://github.com/victor-shepardson/alpha-GAN	2	51	17
2017	7	3D-IWGAN	Improved Adversarial Systems for 3D Object Generation and Reconstruction	https://arxiv.org/abs/1707.09557	https://github.com/EdwardSmith1884/3D-IWGAN	7	70	23
2017	7	AE-GAN	AE-GAN: adversarial eliminating with GAN	https://arxiv.org/abs/1707.05474	-			
2017	7	AlignGAN	AlignGAN: Learning to Align Cross-Domain Images with Conditional Generative Adversarial Networks	https://arxiv.org/abs/1707.01400	-			
2017	7	APE-GAN	APE-GAN: Adversarial Perturbation Elimination with GAN	https://arxiv.org/abs/1707.05474	-			
2017	7	ARDA	Adversarial Representation Learning for Domain Adaptation	https://arxiv.org/abs/1707.01217	-			
2017	7	DAN	Distributional Adversarial Networks	https://arxiv.org/abs/1706.09549	-			
2017	7	l-GAN	Representation Learning and Adversarial Generation of 3D Point Clouds	https://arxiv.org/abs/1707.02392	-			
2017	7	LD-GAN	Linear Discriminant Generative Adversarial Networks	https://arxiv.org/abs/1707.07831	-			
2017	7	LeGAN	Likelihood Estimation for Generative Adversarial Networks	https://arxiv.org/abs/1707.07530	-			
2017	7	MMGAN	MMGAN: Manifold Matching Generative Adversarial Network for Generating Images	https://arxiv.org/abs/1707.08273	-			
2017	7	MoCoGAN	MoCoGAN: Decomposing Motion and Content for Video Generation	https://arxiv.org/abs/1707.04993	https://github.com/sergeytulyakov/mocogan	17	180	47
2017	7	ResGAN	Generative Adversarial Network based on Resnet for Conditional Image Restoration	https://arxiv.org/abs/1707.04881	-			
2017	7	SisGAN	Semantic Image Synthesis via Adversarial Learning	https://arxiv.org/abs/1707.06873	-			
2017	7	ss-InfoGAN	Guiding InfoGAN with Semi-Supervision	https://arxiv.org/abs/1707.04487	-			
2017	7	SSGAN	SSGAN: Secure Steganography Based on Generative Adversarial Networks	https://arxiv.org/abs/1707.01613	-			
2017	7	SteinGAN	Learning Deep Energy Models: Contrastive Divergence vs. Amortized MLE	https://arxiv.org/abs/1707.00797	-			
2017	7	VRAL	Variance Regularizing Adversarial Learning	https://arxiv.org/abs/1707.00309	-			
2017	8	3D-RecGAN	3D Object Reconstruction from a Single Depth View with Adversarial Learning	https://arxiv.org/abs/1708.07969	https://github.com/Yang7879/3D-RecGAN	8	65	27
2017	8	ABC-GAN	ABC-GAN: Adaptive Blur and Control for improved training stability of Generative Adversarial Networks	https://drive.google.com/file/d/0B3wEP_lEl0laVTdGcHE2VnRiMlE/view	https://github.com/IgorSusmelj/ABC-GAN	1	4	1
2017	8	ASDL-GAN	Automatic Steganographic Distortion Learning Using a Generative Adversarial Network	https://ieeexplore.ieee.org/document/8017430/	-			
2017	8	BGAN	Binary Generative Adversarial Networks for Image Retrieval	https://arxiv.org/abs/1708.04150	https://github.com/htconquer/BGAN	6	16	15
2017	8	CDcGAN	Simultaneously Color-Depth Super-Resolution with Conditional Generative Adversarial Network	https://arxiv.org/abs/1708.09105	-			
2017	8	CGAN	Controllable Generative Adversarial Network	https://arxiv.org/abs/1708.00598	-			
2017	8	constrast-GAN	Generative Semantic Manipulation with Contrasting GAN	https://arxiv.org/abs/1708.00315	-			
2017	8	Coulomb GAN	Coulomb GANs: Provably Optimal Nash Equilibria via Potential Fields	https://arxiv.org/abs/1708.08819	-			
2017	8	DM-GAN	Dual Motion GAN for Future-Flow Embedded Video Prediction	https://arxiv.org/abs/1708.00284	-			
2017	8	GAN-sep	GANs for Biological Image Synthesis	https://arxiv.org/abs/1708.04692	https://github.com/aosokin/biogans	5	82	12
2017	8	GAN-VFS	Generative Adversarial Network-based Synthesis of Visible Faces from Polarimetric Thermal Faces	https://arxiv.org/abs/1708.02681	-			
2017	8	MGGAN	Multi-Generator Generative Adversarial Nets	https://arxiv.org/abs/1708.02556	-			
2017	8	PGAN	Probabilistic Generative Adversarial Networks	https://arxiv.org/abs/1708.01886	-			
2017	8	SN-GAN	Spectral Normalization for Generative Adversarial Networks	https://drive.google.com/file/d/0B8HZ50DPgR3eSVV6YlF3XzQxSjQ/view	https://github.com/pfnet-research/chainer-gan-lib	33	297	62
2017	8	SS-GAN	Semi-supervised Conditional GANs	https://arxiv.org/abs/1708.05789	-			
2017	8	VIGAN	VIGAN: Missing View Imputation with Generative Adversarial Networks	https://arxiv.org/abs/1708.06724	-			
2017	9	ARIGAN	ARIGAN: Synthetic Arabidopsis Plants using Generative Adversarial Network	https://arxiv.org/abs/1709.00938	-			
2017	9	CausalGAN	CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training	https://arxiv.org/abs/1709.02023	-			
2017	9	D2GAN	Dual Discriminator Generative Adversarial Nets	http://arxiv.org/abs/1709.03831	-			
2017	9	ExposureGAN	Exposure: A White-Box Photo Post-Processing Framework	https://arxiv.org/abs/1709.09602	https://github.com/yuanming-hu/exposure	12	205	43
2017	9	ExprGAN	ExprGAN: Facial Expression Editing with Controllable Expression Intensity	https://arxiv.org/abs/1709.03842	-			
2017	9	GAMN	Generative Adversarial Mapping Networks	https://arxiv.org/abs/1709.09820	-			
2017	9	GraspGAN	Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping	https://arxiv.org/abs/1709.07857	-			
2017	9	LDAN	Label Denoising Adversarial Network (LDAN) for Inverse Lighting of Face Images	https://arxiv.org/abs/1709.01993	-			
2017	9	LeakGAN	Long Text Generation via Adversarial Training with Leaked Information	https://arxiv.org/abs/1709.08624	-			
2017	9	MD-GAN	Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic Generative Adversarial Networks	https://arxiv.org/abs/1709.07592	-			
2017	9	MuseGAN	MuseGAN: Symbolic-domain Music Generation and Accompaniment with Multi-track Sequential Generative Adversarial Networks	https://arxiv.org/abs/1709.06298	-			
2017	9	OptionGAN	OptionGAN: Learning Joint Reward-Policy Options using Generative Adversarial Inverse Reinforcement Learning	https://arxiv.org/abs/1709.06683	-			
2017	9	PassGAN	PassGAN: A Deep Learning Approach for Password Guessing	https://arxiv.org/abs/1709.00440	-			
2017	9	RefineGAN	Compressed Sensing MRI Reconstruction with Cyclic Loss in Generative Adversarial Networks	https://arxiv.org/abs/1709.00753	-			
2017	9	Splitting GAN	Class-Splitting Generative Adversarial Networks	https://arxiv.org/abs/1709.07359	-			
2017	9	Δ-GAN	Triangle Generative Adversarial Networks	https://arxiv.org/abs/1709.06548	-			
2017	10	CM-GAN	CM-GANs: Cross-modal Generative Adversarial Networks for Common Representation Learning	https://arxiv.org/abs/1710.05106	-			
2017	10	GAN-ATV	A Novel Approach to Artistic Textual Visualization via GAN	https://arxiv.org/abs/1710.10553	-			
2017	10	GAP	Context-Aware Generative Adversarial Privacy	https://arxiv.org/abs/1710.09549	-			
2017	10	GP-GAN	GP-GAN: Gender Preserving GAN for Synthesizing Faces from Landmarks	https://arxiv.org/abs/1710.00962	-			
2017	10	Progressive GAN	"Progressive Growing of GANs for Improved Quality, Stability, and Variation"	https://arxiv.org/abs/1710.10196	https://github.com/tkarras/progressive_growing_of_gans	232	3146	529
2017	10	PS²-GAN	High-Quality Facial Photo-Sketch Synthesis Using Multi-Adversarial Networks	https://arxiv.org/abs/1710.10182	-			
2017	10	SVSGAN	SVSGAN: Singing Voice Separation via Generative Adversarial Network	https://arxiv.org/abs/1710.11428	-			
2017	10	TGAN	Tensorizing Generative Adversarial Nets	https://arxiv.org/abs/1710.10772	-			
2017	11	3D-ED-GAN	Shape Inpainting using 3D Generative Adversarial Network and Recurrent Convolutional Networks	https://arxiv.org/abs/1711.06375	-			
2017	11	ABC-GAN	GANs for LIFE: Generative Adversarial Networks for Likelihood Free Inference	https://arxiv.org/abs/1711.11139	-			
2017	11	ACtuAL	ACtuAL: Actor-Critic Under Adversarial Learning	https://arxiv.org/abs/1711.04755	-			
2017	11	AttGAN	Arbitrary Facial Attribute Editing: Only Change What You Want	https://arxiv.org/abs/1711.10678	https://github.com/LynnHo/AttGAN-Tensorflow	11	158	19
2017	11	AttnGAN	AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks	https://arxiv.org/abs/1711.10485	https://github.com/taoxugit/AttnGAN	27	420	97
2017	11	BCGAN	Bidirectional Conditional Generative Adversarial networks	https://arxiv.org/abs/1711.07461	-			
2017	11	BicycleGAN	Toward Multimodal Image-to-Image Translation	https://arxiv.org/abs/1711.11586	https://github.com/junyanz/BicycleGAN	30	696	113
2017	11	CatGAN	CatGAN: Coupled Adversarial Transfer for Domain Generation	https://arxiv.org/abs/1711.08904	-			
2017	11	CoAtt-GAN	Are You Talking to Me? Reasoned Visual Dialog Generation through Adversarial Learning	https://arxiv.org/abs/1711.07613	-			
2017	11	ConceptGAN	Learning Compositional Visual Concepts with Mutual Consistency	https://arxiv.org/abs/1711.06148	-			
2017	11	Cover-GAN	Generative Steganography with Kerckhoffs' Principle based on Generative Adversarial Networks	https://arxiv.org/abs/1711.04916	-			
2017	11	D-GAN	Differential Generative Adversarial Networks: Synthesizing Non-linear Facial Variations with Limited Number of Training Data	https://arxiv.org/abs/1711.10267	-			
2017	11	DAGAN	Data Augmentation Generative Adversarial Networks	https://arxiv.org/abs/1711.04340	-			
2017	11	DeblurGAN	DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks	https://arxiv.org/abs/1711.07064	https://github.com/KupynOrest/DeblurGAN	35	971	204
2017	11	DNA-GAN	DNA-GAN: Learning Disentangled Representations from Multi-Attribute Images	https://arxiv.org/abs/1711.05415	-			
2017	11	DRPAN	Discriminative Region Proposal Adversarial Networks for High-Quality Image-to-Image Translation	https://arxiv.org/abs/1711.09554	-			
2017	11	FIGAN	Frame Interpolation with Multi-Scale Deep Loss Functions and Generative Adversarial Networks	https://arxiv.org/abs/1711.06045	-			
2017	11	FSEGAN	Exploring Speech Enhancement with Generative Adversarial Networks for Robust Speech Recognition	https://arxiv.org/abs/1711.05747	-			
2017	11	FTGAN	Hierarchical Video Generation from Orthogonal Information: Optical Flow and Texture	https://arxiv.org/abs/1711.09618	-			
2017	11	GANDI	Guiding the search in continuous state-action spaces by learning an action sampling distribution from off-target samples	https://arxiv.org/abs/1711.01391	-			
2017	11	GPU	A generative adversarial framework for positive-unlabeled classification	https://arxiv.org/abs/1711.08054	-			
2017	11	HAN	Chinese Typeface Transformation with Hierarchical Adversarial Network	https://arxiv.org/abs/1711.06448	-			
2017	11	HP-GAN	HP-GAN: Probabilistic 3D human motion prediction via GAN	https://arxiv.org/abs/1711.09561	-			
2017	11	HR-DCGAN	High-Resolution Deep Convolutional Generative Adversarial Networks	https://arxiv.org/abs/1711.06491	-			
2017	11	IFcVAEGAN	Conditional Autoencoders with Adversarial Information Factorization	https://arxiv.org/abs/1711.05175	-			
2017	11	In2I	In2I : Unsupervised Multi-Image-to-Image Translation Using Generative Adversarial Networks	https://arxiv.org/abs/1711.09334	-			
2017	11	Iterative-GAN	Two Birds with One Stone: Iteratively Learn Facial Attributes with GANs	https://arxiv.org/abs/1711.06078	https://github.com/punkcure/Iterative-GAN	1	8	4
2017	11	IVE-GAN	IVE-GAN: Invariant Encoding Generative Adversarial Networks	https://arxiv.org/abs/1711.08646	-			
2017	11	iVGAN	Towards an Understanding of Our World by GANing Videos in the Wild	https://arxiv.org/abs/1711.11453	https://github.com/bernhard2202/improved-video-gan	7	290	28
2017	11	KBGAN	KBGAN: Adversarial Learning for Knowledge Graph Embeddings	https://arxiv.org/abs/1711.04071	-			
2017	11	KGAN	KGAN: How to Break The Minimax Game in GAN	https://arxiv.org/abs/1711.01744	-			
2017	11	LGAN	Global versus Localized Generative Adversarial Nets	https://arxiv.org/abs/1711.06020	-			
2017	11	MLGAN	Metric Learning-based Generative Adversarial Network	https://arxiv.org/abs/1711.02792	-			
2017	11	ORGAN	3D Reconstruction of Incomplete Archaeological Objects Using a Generative Adversary Network	https://arxiv.org/abs/1711.06363	-			
2017	11	Pip-GAN	Pipeline Generative Adversarial Networks for Facial Images Generation with Multiple Attributes	https://arxiv.org/abs/1711.10742	-			
2017	11	pix2pixHD	High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs	https://arxiv.org/abs/1711.11585	https://github.com/NVIDIA/pix2pixHD	116	2426	424
2017	11	Sobolev GAN	Sobolev GAN	https://arxiv.org/abs/1711.04894	-			
2017	11	StarGAN	StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation	https://arxiv.org/abs/1711.09020	https://github.com/yunjey/StarGAN	109	3047	491
2017	11	TGAN	Tensor-Generative Adversarial Network with Two-dimensional Sparse Coding: Application to Real-time Indoor Localization	https://arxiv.org/abs/1711.02666	-			
2017	11	tripletGAN	TripletGAN: Training Generative Model with Triplet Loss	https://arxiv.org/abs/1711.05084	-			
2017	11	VA-GAN	Visual Feature Attribution using Wasserstein GANs	https://arxiv.org/abs/1711.08998	-			
2017	11	XGAN	XGAN: Unsupervised Image-to-Image Translation for many-to-many Mappings	https://arxiv.org/abs/1711.05139	-			
2017	11	ZipNet-GAN	ZipNet-GAN: Inferring Fine-grained Mobile Traffic Patterns via a Generative Adversarial Neural Network	https://arxiv.org/abs/1711.02413	-			
2017	12	ACGAN	Coverless Information Hiding Based on Generative adversarial networks	https://arxiv.org/abs/1712.06951	-			
2017	12	CA-GAN	Composition-aided Sketch-realistic Portrait Generation	https://arxiv.org/abs/1712.00899	-			
2017	12	ComboGAN	ComboGAN: Unrestrained Scalability for Image Domain Translation	https://arxiv.org/abs/1712.06909	https://github.com/AAnoosheh/ComboGAN	5	84	15
2017	12	DF-GAN	Learning Disentangling and Fusing Networks for Face Completion Under Structured Occlusions	https://arxiv.org/abs/1712.04646	-			
2017	12	Dynamics Transfer GAN	Dynamics Transfer GAN: Generating Video by Transferring Arbitrary Temporal Dynamics from a Source Video to a Single Target Image	https://arxiv.org/abs/1712.03534	-			
2017	12	EnergyWGAN	Energy-relaxed Wassertein GANs (EnergyWGAN): Towards More Stable and High Resolution Image Generation	https://arxiv.org/abs/1712.01026	-			
2017	12	ExGAN	Eye In-Painting with Exemplar Generative Adversarial Networks	https://arxiv.org/abs/1712.03999	-			
2017	12	f-CLSWGAN	Feature Generating Networks for Zero-Shot Learning	https://arxiv.org/abs/1712.00981	-			
2017	12	FusionGAN	Learning to Fuse Music Genres with Generative Adversarial Dual Learning	https://arxiv.org/abs/1712.01456	-			
2017	12	G2-GAN	Geometry Guided Adversarial Facial Expression Synthesis	https://arxiv.org/abs/1712.03474	-			
2017	12	GAGAN	GAGAN: Geometry-Aware Generative Adverserial Networks	https://arxiv.org/abs/1712.00684	-			
2017	12	GAN-RS	Towards Qualitative Advancement of Underwater Machine Vision with Generative Adversarial Networks	https://arxiv.org/abs/1712.00736	-			
2017	12	GANG	GANGs: Generative Adversarial Network Games	https://arxiv.org/abs/1712.00679	-			
2017	12	GANosaic	GANosaic: Mosaic Creation with Generative Texture Manifolds	https://arxiv.org/abs/1712.00269	-			
2017	12	IdCycleGAN	Face Translation between Images and Videos using Identity-aware CycleGAN	https://arxiv.org/abs/1712.00971	-			
2017	12	manifold-WGAN	Manifold-valued Image Generation with Wasserstein Adversarial Networks	https://arxiv.org/abs/1712.01551	-			
2017	12	MC-GAN	Multi-Content GAN for Few-Shot Font Style Transfer	https://arxiv.org/abs/1712.00516	https://github.com/azadis/MC-GAN	9	167	54
2017	12	MIL-GAN	Multimodal Storytelling via Generative Adversarial Imitation Learning	https://arxiv.org/abs/1712.01455	-			
2017	12	MS-GAN	Temporal Coherency based Criteria for Predicting Video Frames using Deep Multi-stage Generative Adversarial Networks	http://papers.nips.cc/paper/7014-temporal-coherency-based-criteria-for-predicting-video-frames-using-deep-multi-stage-generative-adversarial-networks	-			
2017	12	PacGAN	PacGAN: The power of two samples in generative adversarial networks	https://arxiv.org/abs/1712.04086	-			
2017	12	PN-GAN	Pose-Normalized Image Generation for Person Re-identification	https://arxiv.org/abs/1712.02225	-			
2017	12	PPAN	Privacy-Preserving Adversarial Networks	https://arxiv.org/abs/1712.07008	-			
2017	12	RAN	RAN4IQA: Restorative Adversarial Nets for No-Reference Image Quality Assessment	https://arxiv.org/abs/1712.05444				
2017	12	SGAN	SGAN: An Alternative Training of Generative Adversarial Networks	https://arxiv.org/abs/1712.02330	-			
2017	12	SRPGAN	SRPGAN: Perceptual Generative Adversarial Network for Single Image Super Resolution	https://arxiv.org/abs/1712.05927	-			
2017	12	ST-CGAN	Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal	https://arxiv.org/abs/1712.02478	-			
2017	12	Super-FAN	Super-FAN: Integrated facial landmark localization and super-resolution of real-world low resolution faces in arbitrary poses with GANs	https://arxiv.org/abs/1712.02765	-			
2017	12	TV-GAN	TV-GAN: Generative Adversarial Network Based Thermal to Visible Face Recognition	https://arxiv.org/abs/1712.02514	-			
2017	12	UGACH	Unsupervised Generative Adversarial Cross-modal Hashing	https://arxiv.org/abs/1712.00358	-			
2017	12	UV-GAN	UV-GAN: Adversarial Facial UV Map Completion for Pose-invariant Face Recognition	https://arxiv.org/abs/1712.04695	-			
2017	12	VGAN	Text Generation Based on Generative Adversarial Nets with Latent Variable	https://arxiv.org/abs/1712.00170	-			
2017	12	weGAN	Generative Adversarial Nets for Multiple Text Corpora	https://arxiv.org/abs/1712.09127	-			
2018	1	AdvGAN	Generating adversarial examples with adversarial networks	https://arxiv.org/abs/1801.02610	-			
2018	1	CFG-GAN	Composite Functional Gradient Learning of Generative Adversarial Models	https://arxiv.org/abs/1801.06309	-			
2018	1	CipherGAN	Unsupervised Cipher Cracking Using Discrete GANs	https://arxiv.org/abs/1801.04883	-			
2018	1	Cross-GAN	Crossing Generative Adversarial Networks for Cross-View Person Re-identification	https://arxiv.org/abs/1801.01760	-			
2018	1	dp-GAN	Differentially Private Releasing via Deep Generative Model	https://arxiv.org/abs/1801.01594	-			
2018	1	ecGAN	eCommerceGAN : A Generative Adversarial Network for E-commerce	https://arxiv.org/abs/1801.03244	-			
2018	1	FusedGAN	Semi-supervised FusedGAN for Conditional Image Generation	https://arxiv.org/abs/1801.05551	-			
2018	1	GeoGAN	Generating Instance Segmentation Annotation by Geometry-guided GAN 	https://arxiv.org/abs/1801.08839 	-			
2018	1	GLCA-GAN	Global and Local Consistent Age Generative Adversarial Networks 	https://arxiv.org/abs/1801.08390	-			
2018	1	LAC-GAN	Grounded Language Understanding for Manipulation Instructions Using GAN-Based Classification	https://arxiv.org/abs/1801.05096	-			
2018	1	MaskGAN	MaskGAN: Better Text Generation via Filling in the ______ 	https://arxiv.org/abs/1801.07736 	-			
2018	1	SG-GAN	Semantic-aware Grad-GAN for Virtual-to-Real Urban Scene Adaption	https://arxiv.org/abs/1801.01726	https://github.com/Peilun-Li/SG-GAN	5	36	8
2018	1	SketchyGAN	SketchyGAN: Towards Diverse and Realistic Sketch to Image Synthesis	https://arxiv.org/abs/1801.02753	-			
2018	1	tempoGAN	"tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow"	https://arxiv.org/abs/1801.09710	-			
2018	1	UGAN	Enhancing Underwater Imagery using Generative Adversarial Networks	https://arxiv.org/abs/1801.04011	-			
2018	2	AmbientGAN	AmbientGAN: Generative models from lossy measurements	https://openreview.net/forum?id=Hy7fDog0b	https://github.com/AshishBora/ambient-gan	2	57	17
2018	2	ATA-GAN	Attention-Aware Generative Adversarial Networks (ATA-GANs)	https://arxiv.org/abs/1802.09070	-			
2018	2	C-GAN 	Face Aging with Contextual Generative Adversarial Nets 	https://arxiv.org/abs/1802.00237 	-			
2018	2	CapsuleGAN	CapsuleGAN: Generative Adversarial Capsule Network 	http://arxiv.org/abs/1802.06167	-			
2018	2	DA-GAN 	DA-GAN: Instance-level Image Translation by Deep Attention Generative Adversarial Networks (with Supplementary Materials)	http://arxiv.org/abs/1802.06454	-			
2018	2	DP-GAN	DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text 	https://arxiv.org/abs/1802.01345 	-			
2018	2	DPGAN 	Differentially Private Generative Adversarial Network 	http://arxiv.org/abs/1802.06739	-			
2018	2	First Order GAN 	First Order Generative Adversarial Networks 	https://arxiv.org/abs/1802.04591	https://github.com/zalandoresearch/first_order_gan	5	23	7
2018	2	GC-GAN	Geometry-Contrastive Generative Adversarial Network for Facial Expression Synthesis	https://arxiv.org/abs/1802.01822 	-			
2018	2	LB-GAN	Load Balanced GANs for Multi-view Face Image Synthesis	http://arxiv.org/abs/1802.07447	-			
2018	2	MAGAN	MAGAN: Aligning Biological Manifolds	https://arxiv.org/abs/1803.00385	-			
2018	2	ND-GAN	Novelty Detection with GAN	https://arxiv.org/abs/1802.10560	-			
2018	2	PGD-GAN	Solving Linear Inverse Problems Using GAN Priors: An Algorithm with Provable Guarantees	https://arxiv.org/abs/1802.08406	-			
2018	2	RadialGAN	RadialGAN: Leveraging multiple datasets to improve target-specific predictive models using Generative Adversarial Networks 	http://arxiv.org/abs/1802.06403	-			
2018	2	SAR-GAN	Generating High Quality Visible Images from SAR Images Using CNNs	https://arxiv.org/abs/1802.10036	-			
2018	2	SCH-GAN	SCH-GAN: Semi-supervised Cross-modal Hashing by Generative Adversarial Network 	https://arxiv.org/abs/1802.02488 	-			
2018	2	StainGAN	StainGAN: Stain Style Transfer for Digital Histological Images	https://arxiv.org/abs/1804.01601	-			
2018	2	SWGAN	Solving Approximate Wasserstein GANs to Stationarity	https://arxiv.org/abs/1802.08249	-			
2018	2	VoiceGAN	Voice Impersonation using Generative Adversarial Networks 	http://arxiv.org/abs/1802.06840	-			
2018	2	WaveGAN	Synthesizing Audio with Generative Adversarial Networks 	https://arxiv.org/abs/1802.04208	-			
2018	3	Attention-GAN	Attention-GAN for Object Transfiguration in Wild Images	https://arxiv.org/abs/1803.06798	-			
2018	3	B-DCGAN	B-DCGAN:Evaluation of Binarized DCGAN for FPGA	https://arxiv.org/abs/1803.10930	-			
2018	3	BAGAN	BAGAN: Data Augmentation with Balancing GAN	https://arxiv.org/abs/1803.09655	-			
2018	3	BranchGAN	Branched Generative Adversarial Networks for Multi-Scale Image Manifold Learning	https://arxiv.org/abs/1803.08467	-			
2018	3	D2IA-GAN	Tagging like Humans: Diverse and Distinct Image Annotation	https://arxiv.org/abs/1804.00113	-			
2018	3	DBLRGAN	Adversarial Spatio-Temporal Learning for Video Deblurring	https://arxiv.org/abs/1804.00533	-			
2018	3	E-GAN	Evolutionary Generative Adversarial Networks	https://arxiv.org/abs/1803.00657	-			
2018	3	ELEGANT	ELEGANT: Exchanging Latent Encodings with GAN for Transferring Multiple Face Attributes	https://arxiv.org/abs/1803.10562	-			
2018	3	Fictitious GAN	Fictitious GAN: Training GANs with Historical Models	https://arxiv.org/abs/1803.08647	-			
2018	3	GAAN	Generative Adversarial Autoencoder Networks	https://arxiv.org/abs/1803.08887	-			
2018	3	GONet	GONet: A Semi-Supervised Deep Learning Approach For Traversability Estimation	https://arxiv.org/abs/1803.03254	-			
2018	3	memoryGAN	Memorization Precedes Generation: Learning Unsupervised GANs with Memory Networks	https://arxiv.org/abs/1803.01500	-			
2018	3	MTGAN	MTGAN: Speaker Verification through Multitasking Triplet Generative Adversarial Networks	https://arxiv.org/abs/1803.09059	-			
2018	3	NCE-GAN	Dihedral angle prediction using generative adversarial networks	https://arxiv.org/abs/1803.10996	-			
2018	3	NetGAN	NetGAN: Generating Graphs via Random Walks	https://arxiv.org/abs/1803.00816	-			
2018	3	OCAN	One-Class Adversarial Nets for Fraud Detection	https://arxiv.org/abs/1803.01798	-			
2018	3	OT-GAN	Improving GANs Using Optimal Transport	https://arxiv.org/abs/1803.05573	-			
2018	3	PGGAN	Patch-Based Image Inpainting with Generative Adversarial Networks	https://arxiv.org/abs/1803.07422	-			
2018	3	Sdf-GAN	Sdf-GAN: Semi-supervised Depth Fusion with Multi-scale Adversarial Networks	https://arxiv.org/abs/1803.06657	-			
2018	3	Social GAN	Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks	https://arxiv.org/abs/1803.10892	-			
2018	3	Spike-GAN	Synthesizing realistic neural population activity patterns using Generative Adversarial Networks	https://arxiv.org/abs/1803.00338	-			
2018	3	ST-GAN	ST-GAN: Spatial Transformer Generative Adversarial Networks for Image Compositing	https://arxiv.org/abs/1803.01837	-			
2018	3	Text2Shape	Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings	https://arxiv.org/abs/1803.08495	-			
2018	3	tiny-GAN	Analysis of Nonautonomous Adversarial Systems	https://arxiv.org/abs/1803.05045	-			
2018	3	VOS-GAN	VOS-GAN: Adversarial Learning of Visual-Temporal Dynamics for Unsupervised Dense Prediction in Videos	https://arxiv.org/abs/1803.09092	-			
2018	4	3D-PhysNet	3D-PhysNet: Learning the Intuitive Physics of Non-Rigid Object Deformations	https://arxiv.org/abs/1805.00328	-			
2018	4	AF-DCGAN	AF-DCGAN: Amplitude Feature Deep Convolutional GAN for Fingerprint Construction in Indoor Localization System	https://arxiv.org/abs/1804.05347	-			
2018	4	BEAM	Boltzmann Encoded Adversarial Machines	https://arxiv.org/abs/1804.08682	-			
2018	4	CorrGAN	Correlated discrete data generation using adversarial training	https://arxiv.org/abs/1804.00925	-			
2018	4	D-WCGAN	I-vector Transformation Using Conditional Generative Adversarial Networks for Short Utterance Speaker Verification	https://arxiv.org/abs/1804.00290	-			
2018	4	Defo-Net	Defo-Net: Learning Body Deformation using Generative Adversarial Networks	https://arxiv.org/abs/1804.05928	-			
2018	4	DSH-GAN	Deep Semantic Hashing with Generative Adversarial Networks	https://arxiv.org/abs/1804.08275	-			
2018	4	DTR-GAN	DTR-GAN: Dilated Temporal Relational Adversarial Network for Video Summarization	https://arxiv.org/abs/1804.11228	-			
2018	4	DVGAN	Human Motion Modeling using DVGANs	https://arxiv.org/abs/1804.10652	-			
2018	4	EAR	Generative Model for Heterogeneous Inference	https://arxiv.org/abs/1804.09858	-			
2018	4	FBGAN	Feedback GAN (FBGAN) for DNA: a Novel Feedback-Loop Architecture for Optimizing Protein Functions	https://arxiv.org/abs/1804.01694	-			
2018	4	FusionGAN	Generating a Fusion Image: One's Identity and Another's Shape	https://arxiv.org/abs/1804.07455	-			
2018	4	Graphical-GAN	Graphical Generative Adversarial Networks	https://arxiv.org/abs/1804.03429	-			
2018	4	IterGAN	IterGANs: Iterative GANs to Learn and Control 3D Object Transformation	https://arxiv.org/abs/1804.05651	-			
2018	4	M-AAE	Mask-aware Photorealistic Face Attribute Manipulation	https://arxiv.org/abs/1804.08882	-			
2018	4	MelanoGAN	MelanoGANs: High Resolution Skin Lesion Synthesis with GANs	https://arxiv.org/abs/1804.04338	-			
2018	4	MGGAN	MGGAN: Solving Mode Collapse using Manifold Guided Training	https://arxiv.org/abs/1804.04391	-			
2018	4	ModularGAN	Modular Generative Adversarial Networks	https://arxiv.org/abs/1804.03343	-			
2018	4	NAN	Understanding Humans in Crowded Scenes: Deep Nested Adversarial Learning and A New Benchmark for Multi-Human Parsing	https://arxiv.org/abs/1804.03287	-			
2018	4	PM-GAN	PM-GANs: Discriminative Representation Learning for Action Recognition Using Partial-modalities	https://arxiv.org/abs/1804.06248	-			
2018	4	ProGanSR	A Fully Progressive Approach to Single-Image Super-Resolution	https://arxiv.org/abs/1804.02900	-			
2018	4	PS-GAN	Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond	https://arxiv.org/abs/1804.02047	-			
2018	4	ReConNN	Reconstruction of Simulation-Based Physical Field with Limited Samples by Reconstruction Neural Network	https://arxiv.org/abs/1805.00528	-			
2018	4	SAGA	Generative Adversarial Learning for Spectrum Sensing	https://arxiv.org/abs/1804.00709	-			
2018	4	sGAN 	Generative Adversarial Training for MRA Image Synthesis Using Multi-Contrast MRI	https://arxiv.org/abs/1804.04366	-			
2018	4	Sketcher-Refiner GAN	Learning Myelin Content in Multiple Sclerosis from Multimodal MRI through Adversarial Training	https://arxiv.org/abs/1804.08039	-			
2018	4	SyncGAN	SyncGAN: Synchronize the Latent Space of Cross-modal Generative Adversarial Networks	https://arxiv.org/abs/1804.00410	-			
2018	4	TGANs-C	To Create What You Tell: Generating Videos from Captions	https://arxiv.org/abs/1804.08264	-			
2018	4	UT-SCA-GAN	Spatial Image Steganography Based on Generative Adversarial Network	https://arxiv.org/abs/1804.07939	-			
2018	5	AdvEntuRe	AdvEntuRe: Adversarial Training for Textual Entailment with Knowledge-Guided Examples	https://arxiv.org/abs/1805.04680	-			
2018	5	AVID	AVID: Adversarial Visual Irregularity Detection	https://arxiv.org/abs/1805.09521	-			
2018	5	BourGAN	BourGAN: Generative Networks with Metric Embeddings	https://arxiv.org/abs/1805.07674	-			
2018	5	BRE	Improving GAN Training via Binarized Representation Entropy (BRE) Regularization	https://arxiv.org/abs/1805.03644	https://github.com/BorealisAI/bre-gan	3	7	2
2018	5	cd-GAN	Conditional Image-to-Image Translation	https://arxiv.org/abs/1805.00251	-			
2018	5	cowboy	Defending Against Adversarial Attacks by Leveraging an Entire GAN	https://arxiv.org/abs/1805.10652	-			
2018	5	CSG	Speech-Driven Expressive Talking Lips with Conditional Sequential Generative Adversarial Networks	https://arxiv.org/abs/1806.00154	-			
2018	5	Defense-GAN	Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models 	https://arxiv.org/abs/1805.06605 	https://github.com/kabkabm/defensegan	2	35	8
2018	5	DialogWAE	DialogWAE: Multimodal Response Generation with Conditional Wasserstein Auto-Encoder	https://arxiv.org/abs/1805.12352	-			
2018	5	DTLC-GAN	Generative Adversarial Image Synthesis with Decision Tree Latent Controller	https://arxiv.org/abs/1805.10603	-			
2018	5	FairGAN	FairGAN: Fairness-aware Generative Adversarial Networks	https://arxiv.org/abs/1805.11202	-			
2018	5	Fairness GAN	Fairness GAN	https://arxiv.org/abs/1805.09910	-			
2018	5	FakeGAN	Detecting Deceptive Reviews using Generative Adversarial Networks	https://arxiv.org/abs/1805.10364	-			
2018	5	FBGAN	Featurized Bidirectional GAN: Adversarial Defense via Adversarially Learned Semantic Inference	https://arxiv.org/abs/1805.07862	-			
2018	5	FC-GAN	Fast-converging Conditional Generative Adversarial Networks for Image Synthesis	https://arxiv.org/abs/1805.01972	-			
2018	5	GAF	Generative Adversarial Forests for Better Conditioned Adversarial Learning	https://arxiv.org/abs/1805.05185	-			
2018	5	GAN Q-learning	GAN Q-learning	https://arxiv.org/abs/1805.04874	-			
2018	5	GAN-SD	Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning	https://arxiv.org/abs/1805.10000	-			
2018	5	GAN-Word2Vec	Adversarial Training of Word2Vec for Basket Completion	https://arxiv.org/abs/1805.08720	-			
2018	5	GANAX	GANAX: A Unified MIMD-SIMD Acceleration for Generative Adversarial Networks	https://arxiv.org/abs/1806.01107	-			
2018	5	GT-GAN	Deep Graph Translation	https://arxiv.org/abs/1805.09980	-			
2018	5	HAN	Bidirectional Learning for Robust Neural Networks	https://arxiv.org/abs/1805.08006	-			
2018	5	HiGAN	Exploiting Images for Video Recognition with Hierarchical Generative Adversarial Networks	https://arxiv.org/abs/1805.04384	-			
2018	5	hredGAN	Multi-turn Dialogue Response Generation in an Adversarial Learning framework	https://arxiv.org/abs/1805.11752	-			
2018	5	MC-GAN	MC-GAN: Multi-conditional Generative Adversarial Network for Image Synthesis	https://arxiv.org/abs/1805.01123	-			
2018	5	MEGAN	MEGAN: Mixture of Experts of Generative Adversarial Networks for Multimodal Image Generation	https://arxiv.org/abs/1805.02481	-			
2018	5	MolGAN	MolGAN: An implicit generative model for small molecular graphs	https://arxiv.org/abs/1805.11973	-			
2018	5	N2RPP	N2RPP: An Adversarial Network to Rebuild Plantar Pressure for ACLD Patients	https://arxiv.org/abs/1805.02825	-			
2018	5	PD-WGAN	Primal-Dual Wasserstein GAN	https://arxiv.org/abs/1805.09575	-			
2018	5	POGAN	Perceptually Optimized Generative Adversarial Network for Single Image Dehazing	https://arxiv.org/abs/1805.01084	-			
2018	5	PSGAN	PSGAN: A Generative Adversarial Network for Remote Sensing Image Pan-Sharpening	https://arxiv.org/abs/1805.03371	-			
2018	5	ReGAN	ReGAN: RE[LAX|BAR|INFORCE] based Sequence Generation using GANs	https://arxiv.org/abs/1805.02788	https://github.com/TalkToTheGAN/REGAN	2	24	2
2018	5	RegCGAN	Unpaired Multi-Domain Image Generation via Regularized Conditional GANs	https://arxiv.org/abs/1805.02456	-			
2018	5	RoCGAN	Robust Conditional Generative Adversarial Networks	https://arxiv.org/abs/1805.08657	-			
2018	5	SAGAN	Self-Attention Generative Adversarial Networks	https://arxiv.org/abs/1805.08318	-			
2018	5	SG-GAN	Sparsely Grouped Multi-task Generative Adversarial Networks for Facial Attribute Manipulation	https://arxiv.org/abs/1805.07509	-			
2018	5	speech-driven animation GAN	End-to-End Speech-Driven Facial Animation with Temporal GANs	https://arxiv.org/abs/1805.09313	-			
2018	5	WGAN-CLS	Text to Image Synthesis Using Generative Adversarial Networks	https://arxiv.org/abs/1805.00676	-			
2018	6	Adaptive GAN	Customizing an Adversarial Example Generator with Class-Conditional GANs	https://arxiv.org/abs/1806.10496	-			
2018	6	APD	Adversarial Distillation of Bayesian Neural Network Posteriors	https://arxiv.org/abs/1806.10317	-			
2018	6	BinGAN	BinGAN: Learning Compact Binary Descriptors with a Regularized GAN	https://arxiv.org/abs/1806.06778	-			
2018	6	BWGAN	Banach Wasserstein GAN	https://arxiv.org/abs/1806.06621	-			
2018	6	CapsGAN	CapsGAN: Using Dynamic Routing for Generative Adversarial Networks	https://arxiv.org/abs/1806.03968	-			
2018	6	CR-GAN	CR-GAN: Learning Complete Representations for Multi-view Generation	https://arxiv.org/abs/1806.11191	-			
2018	6	DMGAN	Disconnected Manifold Learning for Generative Adversarial Networks	https://arxiv.org/abs/1806.00880	-			
2018	6	EL-GAN	EL-GAN: Embedding Loss Driven Generative Adversarial Networks for Lane Detection	https://arxiv.org/abs/1806.05525	-			
2018	6	FrankenGAN	rankenGAN: Guided Detail Synthesis for Building Mass-Models Using Style-Synchonized GANs	https://arxiv.org/abs/1806.07179	-			
2018	6	GAIN 	GAIN: Missing Data Imputation using Generative Adversarial Nets	https://arxiv.org/abs/1806.02920	-			
2018	6	GANG	Beyond Local Nash Equilibria for Adversarial Networks	https://arxiv.org/abs/1806.07268	-			
2018	6	GATS	Sample-Efficient Deep RL with Generative Adversarial Tree Search	https://arxiv.org/abs/1806.05780	-			
2018	6	IR2VI	IR2VI: Enhanced Night Environmental Perception by Unsupervised Thermal Image Translation	https://arxiv.org/abs/1806.09565	-			
2018	6	IRGAN	Generative Adversarial Nets for Information Retrieval: Fundamentals and Advances	https://arxiv.org/abs/1806.03577	-			
2018	6	JointGAN	JointGAN: Multi-Domain Joint Distribution Learning with Generative Adversarial Nets	https://arxiv.org/abs/1806.02978	-			
2018	6	JR-GAN	JR-GAN: Jacobian Regularization for Generative Adversarial Networks	https://arxiv.org/abs/1806.09235	-			
2018	6	LCC-GAN	Adversarial Learning with Local Coordinate Coding	https://arxiv.org/abs/1806.04895	-			
2018	6	MedGAN	MedGAN: Medical Image Translation using GANs	https://arxiv.org/abs/1806.06397	-			
2018	6	MMC-GAN	A Multimodal Classifier Generative Adversarial Network for Carry and Place Tasks from Ambiguous Language Instructions	https://arxiv.org/abs/1806.03847	-			
2018	6	Modified GAN-CLS	Generate the corresponding Image from Text Description using Modified GAN-CLS Algorithm	https://arxiv.org/abs/1806.11302	-			
2018	6	PP-GAN	Privacy-Protective-GAN for Face De-identification	https://arxiv.org/abs/1806.08906	-			
2018	6	SeUDA	Semantic-Aware Generative Adversarial Nets for Unsupervised Domain Adaptation in Chest X-ray Segmentation	https://arxiv.org/abs/1806.00600	-			
2018	6	SN-DCGAN	Generative Adversarial Networks for Unsupervised Object Co-localization	https://arxiv.org/abs/1806.00236	-			
2018	6	SN-PatchGAN	Free-Form Image Inpainting with Gated Convolution	https://arxiv.org/abs/1806.03589	-			
2018	6	SoPhie	SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints	https://arxiv.org/abs/1806.01482	-			
2018	6	SR-CNN-VAE-GAN	Semi-Recurrent CNN-based VAE-GAN for Sequential Data Generation	https://arxiv.org/abs/1806.00509	https://github.com/makbari7/SR-CNN-VAE-GAN	2	8	3
2018	6	StarGAN-VC	StarGAN-VC: Non-parallel many-to-many voice conversion with star generative adversarial networks 	https://arxiv.org/abs/1806.02169	-			
2018	6	table-GAN	Data Synthesis based on Generative Adversarial Networks	https://arxiv.org/abs/1806.03384	-			
2018	6	tcGAN	Cross-modal Hallucination for Few-shot Fine-grained Recognition	https://arxiv.org/abs/1806.05147	-			
2018	6	TD-GAN	Task Driven Generative Modeling for Unsupervised Domain Adaptation: Application to X-ray Image Segmentation	https://arxiv.org/abs/1806.07201	-			
2018	6	tempCycleGAN	Improving Surgical Training Phantoms by Hyperrealism: Deep Unpaired Image-to-Image Translation from Real Surgeries	https://arxiv.org/abs/1806.03627	-			
2018	6	VAC+GAN 	"Versatile Auxiliary Classifier with Generative Adversarial Network (VAC+GAN), Multi Class Scenarios"	https://arxiv.org/abs/1806.07751	-			
2018	7	acGAN	On-line Adaptative Curriculum Learning for GANs	https://arxiv.org/abs/1808.00020	-			
2018	7	AlphaGAN	AlphaGAN: Generative adversarial networks for natural image matting	https://arxiv.org/abs/1807.10088	-			
2018	7	AMC-GAN	Video Prediction with Appearance and Motion Conditions	https://arxiv.org/abs/1807.02635	-			
2018	7	CE-GAN	Deep Learning for Imbalance Data Classification using Class Expert Generative Adversarial Network	https://arxiv.org/abs/1807.04585	-			
2018	7	ciGAN	Conditional Infilling GANs for Data Augmentation in Mammogram Classification	https://arxiv.org/abs/1807.08093	-			
2018	7	CT-GAN	CT-GAN: Conditional Transformation Generative Adversarial Network for Image Attribute Modification	https://arxiv.org/abs/1807.04812	-			
2018	7	DE-GAN	Generative Adversarial Networks with Decoder-Encoder Output Noise	https://arxiv.org/abs/1807.03923	-			
2018	7	Dropout-GAN	Dropout-GAN: Learning from a Dynamic Ensemble of Discriminators	https://arxiv.org/abs/1807.11346	-			
2018	7	Editable GAN	Editable Generative Adversarial Networks: Generating and Editing Faces Simultaneously	https://arxiv.org/abs/1807.07700	-			
2018	7	FGGAN	Adversarial Learning for Fine-grained Image Search	https://arxiv.org/abs/1807.02247	-			
2018	7	GAIA	Generative adversarial interpolative autoencoding: adversarial training on latent space interpolations encourage convex latent distributions	https://arxiv.org/abs/1807.06650	-			
2018	7	GAP	Generative Adversarial Privacy	https://arxiv.org/abs/1807.05306	-			
2018	7	IntroVAE	IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis	https://arxiv.org/abs/1807.06358	-			
2018	7	ISGAN	Invisible Steganography via Generative Adversarial Network	https://arxiv.org/abs/1807.08571	-			
2018	7	LBT	Learning Implicit Generative Models by Teaching Explicit Ones	https://arxiv.org/abs/1807.03870	-			
2018	7	Lipizzaner	Towards Distributed Coevolutionary GANs	https://arxiv.org/abs/1807.08194	-			
2018	7	MIXGAN	MIXGAN: Learning Concepts from Different Domains for Mixture Generation	https://arxiv.org/abs/1807.01659	-			
2018	7	PIONEER	Pioneer Networks: Progressively Growing Generative Autoencoder	https://arxiv.org/abs/1807.03026	-			
2018	7	RaGAN	The relativistic discriminator: a key element missing from standard GAN	https://arxiv.org/abs/1807.00734	-			
2018	7	Resembled GAN	Resembled Generative Adversarial Networks: Two Domains with Similar Attributes	https://arxiv.org/abs/1807.00947	-			
2018	7	sAOG	Deep Structured Generative Models	https://arxiv.org/abs/1807.03877	-			
2018	7	Sem-GAN	Sem-GAN: Semantically-Consistent Image-to-Image Translation	https://arxiv.org/abs/1807.04409	-			
2018	7	SGAN	CT Image Enhancement Using Stacked Generative Adversarial Networks and Transfer Learning for Lesion Segmentation Improvement	https://arxiv.org/abs/1807.07144	-			
2018	7	SiGAN	SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face Hallucination	https://arxiv.org/abs/1807.08370	-			
2018	7	TequilaGAN	TequilaGAN: How to easily identify GAN samples	https://arxiv.org/abs/1807.04919	-			
2018	7	WGAN-L1	Subsampled Turbulence Removal Network	https://arxiv.org/abs/1807.04418	-			
2018	8	BEGAN-CS	Escaping from Collapsing Modes in a Constrained Space	https://arxiv.org/abs/1808.07258	-			
2018	8	Bellman GAN	Distributional Multivariate Policy Evaluation and Exploration with the Bellman GAN	https://arxiv.org/abs/1808.01960	-			
2018	8	BridgeGAN	Generative Adversarial Frontal View to Bird View Synthesis	https://arxiv.org/abs/1808.00327	-			
2018	8	DOPING	DOPING: Generative Data Augmentation for Unsupervised Anomaly Detection with GAN	https://arxiv.org/abs/1808.07632	-			
2018	8	GIN	Generative Invertible Networks (GIN): Pathophysiology-Interpretable Feature Mapping and Virtual Patient Generation	https://arxiv.org/abs/1808.04495	-			
2018	8	GM-GAN	"Gaussian Mixture Generative Adversarial Networks for Diverse Datasets, and the Unsupervised Clustering of Images"	https://arxiv.org/abs/1808.10356	-			
2018	8	ISP-GPM	Inner Space Preserving Generative Pose Machine	https://arxiv.org/abs/1808.02104	-			
2018	8	MinLGAN	Anomaly Detection via Minimum Likelihood Generative Adversarial Networks	https://arxiv.org/abs/1808.00200	-			
2018	8	Recycle-GAN	Recycle-GAN: Unsupervised Video Retargeting	https://arxiv.org/abs/1808.05174	-			
2018	8	ScarGAN	ScarGAN: Chained Generative Adversarial Networks to Simulate Pathological Tissue on Cardiovascular MR Scans	https://arxiv.org/abs/1808.04500	-			
2018	8	Skip-Thought GAN	Generating Text through Adversarial Training using Skip-Thought Vectors	https://arxiv.org/abs/1808.08703	-			
2018	8	StepGAN	Improving Conditional Sequence Generative Adversarial Networks by Stepwise Evaluation	https://arxiv.org/abs/1808.05599	-			
2018	8	T2Net	T2Net: Synthetic-to-Realistic Translation for Solving Single-Image Depth Estimation Tasks	https://arxiv.org/abs/1808.01454	-			
2018	8	TreeGAN	TreeGAN: Syntax-Aware Sequence Generation with Generative Adversarial Networks	https://arxiv.org/abs/1808.07582	-			
2018	8	X-GANs	X-GANs: Image Reconstruction Made Easy for Extreme Cases	https://arxiv.org/abs/1808.04432	-			
2018	9	AE-OT	Latent Space Optimal Transport for Generative Models	https://arxiv.org/abs/1809.05964	-			
2018	9	AIM	Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization	https://arxiv.org/abs/1809.05972	-			
2018	9	Bi-GAN	Autonomously and Simultaneously Refining Deep Neural Network Parameters by a Bi-Generative Adversarial Network Aided Genetic Algorithm	https://arxiv.org/abs/1809.10244	-			
2018	9	BubGAN	BubGAN: Bubble Generative Adversarial Networks for Synthesizing Realistic Bubbly Flow Images	https://arxiv.org/abs/1809.02266	-			
2018	9	CinCGAN	Unsupervised Image Super-Resolution using Cycle-in-Cycle Generative Adversarial Networks	https://arxiv.org/abs/1809.00437	-			
2018	9	ClusterGAN	ClusterGAN : Latent Space Clustering in Generative Adversarial Networks	https://arxiv.org/abs/1809.03627	-			
2018	9	DADA	DADA: Deep Adversarial Data Augmentation for Extremely Low Data Regime Classification	https://arxiv.org/abs/1809.00981	-			
2018	9	DeepFD	Learning to Detect Fake Face Images in the Wild	https://arxiv.org/abs/1809.08754	-			
2018	9	ESRGAN	ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks	https://arxiv.org/abs/1809.00219	-			
2018	9	GAN Lab	GAN Lab: Understanding Complex Deep Generative Models using Interactive Visual Experimentation	https://arxiv.org/abs/1809.01587	-			
2018	9	GAN-AD	Anomaly Detection with Generative Adversarial Networks for Multivariate Time Series	https://arxiv.org/abs/1809.04758	-			
2018	9	GANVO	GANVO: Unsupervised Deep Monocular Visual Odometry and Depth Estimation with Generative Adversarial Networks	https://arxiv.org/abs/1809.05786	-			
2018	9	GcGAN	Geometry-Consistent Adversarial Networks for One-Sided Unsupervised Domain Mapping	https://arxiv.org/abs/1809.05852	-			
2018	9	GraphSGAN	Semi-supervised Learning on Graphs with Generative Adversarial Nets	https://arxiv.org/abs/1809.00130	-			
2018	9	IGMM-GAN	Coupled IGMM-GANs for deep multimodal anomaly detection in human mobility data	https://arxiv.org/abs/1809.02728	-			
2018	9	MeRGAN	Memory Replay GANs: learning to generate images from new categories without forgetting	https://arxiv.org/abs/1809.02058	-			
2018	9	SAM	Sample-Efficient Imitation Learning via Generative Adversarial Nets	https://arxiv.org/abs/1809.02064	-			
2018	9	SiftingGAN	SiftingGAN: Generating and Sifting Labeled Samples to Improve the Remote Sensing Image Scene Classification Baseline in vitro	https://arxiv.org/abs/1809.04985	-			
2018	9	SLSR	Sparse Label Smoothing for Semi-supervised Person Re-Identification	https://arxiv.org/abs/1809.04976	-			
2018	9	Twin-GAN	Twin-GAN -- Unpaired Cross-Domain Image Translation with Weight-Sharing GANs	https://arxiv.org/abs/1809.00946	-			
2018	9	WaveletGLCA-GAN	Global and Local Consistent Wavelet-domain Age Synthesis	https://arxiv.org/abs/1809.07764	-			

## mlp singer
https://github.com/neosapience/mlp-singer.git

## datasets
[celeba](https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip)

##  Implemented GANs

| Name| Venue | Architecture | G_type*| D_type*| Loss | EMA**|
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| [**DCGAN**](https://arxiv.org/abs/1511.06434) | arXiv' 15 | CNN/ResNet*** | N/A | N/A | Vanilla | False |
| [**LSGAN**](https://arxiv.org/abs/1611.04076) | ICCV' 17 | CNN/ResNet*** | N/A | N/A | Least Sqaure | False |
| [**GGAN**](https://arxiv.org/abs/1705.02894) | arXiv' 17 | CNN/ResNet*** | N/A | N/A | Hinge | False |
| [**WGAN-WC**](https://arxiv.org/abs/1701.04862) | ICLR' 17 |  ResNet | N/A | N/A | Wasserstein | False |
| [**WGAN-GP**](https://arxiv.org/abs/1704.00028) | NIPS' 17 |  ResNet | N/A | N/A | Wasserstein |  False |
| [**WGAN-DRA**](https://arxiv.org/abs/1705.07215) | arXiv' 17 |  ResNet | N/A | N/A | Wasserstein | False |
| [**ACGAN**](https://arxiv.org/abs/1610.09585) | ICML' 17 |  ResNet | cBN | AC | Hinge | False |
| [**ProjGAN**](https://arxiv.org/abs/1802.05637) | ICLR' 18 |  ResNet | cBN | PD | Hinge | False |
| [**SNGAN**](https://arxiv.org/abs/1802.05957) | ICLR' 18 |  ResNet | cBN | PD | Hinge | False |
| [**SAGAN**](https://arxiv.org/abs/1805.08318) | ICML' 19 |  ResNet | cBN | PD | Hinge | False |
| [**BigGAN**](https://arxiv.org/abs/1809.11096) | ICLR' 18 |  Big ResNet | cBN | PD | Hinge | True |
| [**BigGAN-Deep**](https://arxiv.org/abs/1809.11096) | ICLR' 18 |  Big ResNet Deep | cBN | PD | Hinge | True |
| [**CRGAN**](https://arxiv.org/abs/1910.12027) | ICLR' 20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**ICRGAN**](https://arxiv.org/abs/2002.04724) | arXiv' 20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**LOGAN**](https://arxiv.org/abs/1912.00953) | arXiv' 19 |  Big ResNet | cBN | PD | Hinge | True |
| [**DiffAugGAN**](https://arxiv.org/abs/2006.10738) | arXiv' 20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**ADAGAN**](https://arxiv.org/abs/2006.06676) | arXiv' 20 |  Big ResNet | cBN | PD/CL | Hinge | True |
| [**ContraGAN**](https://arxiv.org/abs/2006.12681) | arXiv' 20 | Big ResNet | cBN | CL | Hinge | True |
| [**FreezeD**](https://arxiv.org/abs/2002.10964) | CVPRW' 20 | - | - | - | - | - |

https://github.com/ahmedfgad/GeneticAlgorithmPython

### Facebook AI Introduces Multiscale Vision Transformers (MViT), A Transformer Architecture For Representation Learning From Visual Data
Quick Read: https://www.marktechpost.com/.../facebook-ai-introduces.../
Paper: https://arxiv.org/abs/2104.11227?
Github: https://github.com/facebookresearch/SlowFast

### Facebook And TU Graz Introduces An Ultra-Compact AI Generator, ‘DONeRF’, Which Is 48x Faster Than NeRF
Quick Read: https://www.marktechpost.com/.../facebook-and-tu-graz.../
Paper:https://arxiv.org/pdf/2103.03231.pdf
Github: https://github.com/facebookresearch/DONERF


## ACGAN
Title: Rebooting ACGAN: Auxiliary Classifier GANs with Stable Training
Paper: https://arxiv.org/abs/2111.01118
GitHub: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
해당 논문은 올해 Neurips 2021에 발표될 예정이며 내용을 정말 간단히 요약하자면 다음과 같습니다.
요약:
본 논문에서는 ACGAN 학습이 불안정한 원인으로 학습 초기에 뻗어나가는 특징맵과 분류자의 부정확한 예측을 주목했습니다. 이를 해결하기 위해 특징맵을 unit hypersphere에 사영하는 간단한 방법을 사용하였고, 이에 Contrastive learning 아이디어를 더해 D2D-CE로스를 제안하였습니다. Softmax 함수 대신 D2D-CE를 사용하는 ReACGAN은 CIFAR10, Tiny-ImageNet, CUB200, ImageNet, AFHQ에서 기존 모델대비 의미있는 성능향상을 보여주었고, 모델 아키텍처, Differentiable augmentations, 적대적 손실 함수, 하이퍼매개변수 변화에 강겅한 특징을 가지고 있습니다. 이를 StudioGAN 라이브러리를 통해 실험적으로 증명하였고, 특히 ACGAN 계열로는 처음으로 2048 배치 이미지넷 생성 실험에 성공했음을 리포트하였습니다.

## Transform pic to anime: AnimeGANv2 Face Portrait v2
github: https://github.com/bryandlee/animegan2-pytorch
huggingface gradio demo: https://huggingface.co/spaces/akhaliq/AnimeGANv2
## HAN2HAN : Hangul Font Generation
한글 폰트 생성 프로젝트를 진행했습니다. 10글자 내외를 입력하면 해당 글씨체의 스타일대로 폰트를 생성합니다. 생성가능한 글자는 상용한글 + 영어알파벳으로 2,420글자입니다. 
https://github.com/MINED30/HAN2HAN
더 많은 예시와 모델 구조 설명이 있습니다. 

## Microsoft Researchers Unlock New Avenues In Image-Generation Research With Manifold Matching Via Metric Learning
Paper: https://arxiv.org/pdf/2106.10777.pdf
Github:https://github.com/dzld00/pytorch-manifold-matching

## AI Researchers Propose ‘GANgealing’: A GAN-Supervised Algorithm That Learns Transformations of Input Images to Bring Them into Better Joint Alignment
Paper: https://arxiv.org/pdf/2112.05143v1.pdf
Github: https://github.com/wpeebles/gangealing

## generative manifold learning!
https://yilundu.github.io/gem/

We show how to capture the manifold of any signal modality (including cross-modal ones), by representing each signal as a neural field. We can then traverse the latent space between signals and generate new samples!

Such an approach enables our latent manifold to capture the underlying structure in different signal modalities, enabling us to inpaint images as well as generate audio or image hallucinations.
### Style-GAN paper out
Code and pre-trained models for StyleGAN - Official TensorFlow Implementation
https://github.com/NVlabs/stylegan
- code: [Puzer/stylergan-encoder](https://github.com/Puzer/stylegan-encoder)
- code: [StackGAN](https://github.com/hanzhanggit/StackGAN)

# License

MIT

