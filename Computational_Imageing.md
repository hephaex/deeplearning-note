# Computational Vision
 Computational Imaging is a the process of indirectly forming images from measurements using algorithms that rely on a significant amount of computing. 
 In contrast to traditional imaging, computational imaging systems involve a tight integration of the sensing system and the computation in order to form the images of interest. 
 The ubiquitous availability of fast computing platforms (such as multi-core CPUs and GPUs), the advances in algorithms and modern sensing hardware is resulting in imaging systems with significantly enhanced capabilities. 
 Computational Imaging systems cover a broad range of applications include computational microscopy,[1] tomographic imaging, MRI, ultrasound imaging, computational photography, Synthetic Aperture Radar (SAR), seismic imaging etc. 
 The integration of the sensing and the computation in computational imaging systems allows for accessing information which was otherwise not possible.
 
## Building advanced RAG includes having the LLM reason about the query, not just the final context.
https://twitter.com/jerryjliu0/status/1734598198759759952
https://github.com/.../que.../query_transform_cookbook.ipynb

## 오브젝트 검출 모델은 주로 「Backbone」(모델의 기초가 되는 부분), 「Neck」(Backbone에서 추출된 특징을 고도로 변환하는 부분), 그리고 「Head」(물체의 카테고리나 위치를 구체적으로 결정하는 부분)의 세 부분으로 이루어져 있습니다.
현재 모델에는 Neck 부분에서 정보를 성공적으로 통합할 수 없다는 문제가 있습니다. 특히 서로 다른 '계층'의 정보를 통합할 때 정보 손실이 발생하는 문제가 있다.
이 문제를 해결하는 새로운 방법으로 "Gather-and-Distribute"(GD) 메커니즘이 제안되었습니다. 이 메커니즘은 서로 다른 계층의 정보를 전반적으로 수집하고 높은 계층에 분산함으로써 정보 통합을 효율적으로 실현합니다. 이것은 넥 부분의 성능을 크게 향상시킵니다.
새로운 물체 검출 아키텍처인 Gold-YOLO는 이 GD 메커니즘을 도입하여 보다 높은 정밀도의 물체 검출을 실현하고 있습니다. 또한 Gold-YOLO는 ImageNet에서 사전 학습을 통해 모델 수렴 속도와 정확도를 크게 향상시킵니다.
기존의 YOLO 모델과 비교해도 Gold-YOLO는 현저한 정밀도를 자랑합니다. 구체적으로 Gold-YOLO-S는 선행 연구의 YOLOv6-3.0-S보다 높은 AP(Average Precision)를 달성했습니다.

## 이미지와 문장을 “그대로” 입출력할 수 있는 생성 모델 「DreamLLM」 중간 표현 사용하지 않고
DreamLLM: Synergistic Multimodal Comprehension and Creation : https://dreamllm.github.io/, Xi'an Jiaotong University외 
"DreamLLM"이라는 새로운 기술은 다른 유사한 기술과 달리 이미지와 텍스트를 그대로 (원시 데이터) 형태로 입력으로 받아 동일한 형식으로 출력합니다. 예를 들어, 유명한 회화 그림과 '이 그림 설명'이라는 지시를 입력하면 DreamLLM은 정확한 설명을 생성할 수 있습니다.
많은 AI 모델은 데이터(특히 이미지)를 처리할 때 이를 일부 중간 표현으로 변환합니다. 그러나 DreamLLM은 이러한 중간 표현을 만드는 대신 데이터를 원형으로 처리합니다. 이 접근법의 장점은 모델이 데이터의 본질을 보다 직접적으로 파악하고 보다 정밀한 결과를 출력할 가능성이 높다는 점입니다.
게다가 DreamLLM은 인터넷 기사 등에서 텍스트와 이미지가 복잡하게 결합되어 있는 데이터라도 잘 이해하고 새로운 데이터를 유사한 형식으로 생성할 수 있습니다.
일반적으로 이러한 복잡한 데이터를 분석하고 생성하는 것은 어렵지만 DreamLLM은 'token'이라는 마크를 사용하여 텍스트에서 이미지가 어디에 배치되어야 하는지 예측합니다.
DreamLLM은 텍스트와 이미지의 이해 능력으로 높은 평가를 받았습니다. 구체적으로는 MMBench와 MM-Vet이라는 텍스트와 이미지의 조합을 평가하는 테스트로 고득점을 획득하고, MS-COCO라는 이미지 생성의 정확도를 측정하는 테스트에서도 낮은 에러율을 기록하고 있습니다.

## Nougat: Neural Optical Understanding for Academic Documents (2308, Meta)
메타 논문. 수식 포함된 PDF -> LaTex/Markdown 문서 변환, 과학 문서를 마크업 언어로 처리하기 위해 광학 문자 인식(OCR) 작업을 수행하는 시각적 변환기 모델. 모델 및 코드도 공개
프로젝트 : https://facebookresearch.github.io/nougat/
논문 : https://arxiv.org/abs/2308.13418
코드 : https://github.com/facebookresearch/nougat
데모 : https://huggingface.co/spaces/ysharma/nougat 
(내용:번역) 과학 지식은 주로 책과 과학 저널에 저장되며, 종종 PDF 형식으로 저장됩니다. 그러나 PDF 형식은 특히 수학적 표현의 경우 의미 정보가 손실되는 문제가 있습니다. 본 논문에서는 과학 문서를 마크업 언어로 처리하기 위해 광학 문자 인식(OCR) 작업을 수행하는 시각적 변환기 모델인 Nougat(학술 문서를 위한 신경 광학 이해)를 제안하고, 새로운 과학 문서 데이터 세트에 대한 모델의 효과를 입증합니다. 제안된 접근 방식은 사람이 읽을 수 있는 문서와 기계가 읽을 수 있는 텍스트 사이의 격차를 해소함으로써 디지털 시대에 과학 지식의 접근성을 향상시킬 수 있는 유망한 솔루션을 제공합니다. 과학 텍스트 인식에 대한 향후 작업을 가속화하기 위해 모델과 코드를 공개합니다.

## This Artificial Intelligence (AI) Paper From South Korea Proposes FFNeRV: 
A Novel Frame-Wise Video Representation Using Frame-Wise Flow Maps And Multi-Resolution Temporal Grids
Paper: https://arxiv.org/pdf/2212.12294.pdf
Github: https://github.com/maincold2/FFNeRV
Project: https://maincold2.github.io/ffnerv/

## Researchers from U Texas and Apple Propose a Novel Transformer-Based Architecture for Global Multi-Object Tracking
Paper: https://arxiv.org/pdf/2203.13250.pdf
Github: https://github.com/xingyizhou/GTR

## A New Study from CMU and Bosch Center for AI Demonstrated a New Transformer Paradigm in Computer Vision
Paper Summary: https://www.marktechpost.com/.../a-new-study-from-cmu.../
Paper: https://arxiv.org/pdf/2201.09792v1.pdf
Github: https://github.com/locuslab/convmixer

## PyTorch implementations of our ICASSP 2021 paper "Real Versus Fake 4K - Authentic Resolution Assessment"
github: https://github.com/rr8shah/TSARA
datasets: https://zenodo.org/record/4526657
comparing models: 
 - FQPath: https://github.com/mahdihosseini/FQPath
 - HVS-MaxPol: https://github.com/mahdihosseini/HVS-MaxPol
 - Synthetic-MaxPol: https://github.com/mahdihosseini/Synthetic-MaxPol
 - LPC-SI: https://ece.uwaterloo.ca/~z70wang/research/lpcsi/
 - GPC: http://helios.mi.parisdescartes.fr/~moisan/sharpness/
 - MLV: https://www.mathworks.com/matlabcentral/fileexchange/49991-maximum-local-variation-mlv-code-for-sharpness-assessment-of-images
 - SPARISH: https://www.mathworks.com/matlabcentral/fileexchange/55106-sparish
 
## ByteDance의 연구원들은 MetaFormer를 소개합니다: 
CUB-200-2011과 NABirds에서 92.3%와 92.7%를 달성하는 미세한 인식을 위한 통합된 메타 프레임워크
빠른 읽기: https://www.marktechpost.com/.../researchers-from.../
종이: https://arxiv.org/pdf/2203.02751v1.pdf
Github: https://github.com/dqshuai/metaformer

## Computer Vision Research, Waymo Researchers Propose Block-NeRF: A Method That Reconstructs Arbitrarily Large Environments Using NeRFs
Quick Read: https://www.marktechpost.com/.../in-a-latest-computer.../
Project: https://waymo.com/research/block-nerf/
Paper: https://arxiv.org/pdf/2202.05263.pdf

## Natural Scenes from a Single Image, https://arxiv.org/pdf/2012.09855.pdf
Project link: https://infinite-nature.github.io/
Code: https://github.com/google-research/go...
Colab demo: https://colab.research.google.com/git...

## satellite-image-deep-learning
https://github.com/robmarkcole/satellite-image-deep-learning

## DeepHarmonization
Project webpage: https://sites.google.com/site/yihsuantsai/research/cvpr17-harmonization 
Contact: Yi-Hsuan Tsai (wasidennis at gmail dot com)
https://github.com/wasidennis/DeepHarmonization

## In a Latest Computer Vision Research, Waymo Researchers Propose Block-NeRF: A Method That Reconstructs Arbitrarily Large Environments Using NeRFs
Quick Read: https://www.marktechpost.com/.../in-a-latest-computer.../
Project: https://waymo.com/research/block-nerf/
Paper: https://arxiv.org/pdf/2202.05263.pdf

## DETReg
Researchers From Tel Aviv University, UC Berkeley and NVIDIA Introduce ‘DETReg’, A Novel Unsupervised AI For Object Detection
Quick Read: https://www.marktechpost.com/.../researchers-from-tel.../
Codes: https://github.com/amirbar/DETReg
Project: https://www.amirbar.net/detreg/
Paper: https://arxiv.org/pdf/2106.04550.pdf

##  SegFormer,
a new semantic segmentation method has been proposed. You can read about it in the link below: https://arxiv.org/abs/2105.15203
The details for training and testing the model for semantic segmentation using SegFormer are available at my GitHub Repository. 
https://github.com/.../Sematic_Segmentation_With_SegFormer 

## DeeoHDR
CCV'18: Deep High Dynamic Range Imaging with Large Foreground Motions
[Deep High Dynamic Range Imaging with Large Foreground Motions](https://arxiv.org/abs/1711.08937), Shangzhe Wu, Jiarui Xu, Yu-Wing Tai, Chi-Keung Tang, in ECCV, 2018. More results can be found on our [project page](https://elliottwu.com/projects/hdr/). 
https://github.com/elliottwu/DeepHDR

## ResTS: Residual Deep interpretable architecture for plant disease detection
https://www.sciencedirect.com/science/article/pii/S2214317321000482/pdfft?md5=6b049240ca4d7569bd16b2b05ce4e247&pid=1-s2.0-S2214317321000482-main.pdf

## General Light Microscopy 
- https://www.ibiology.org/online-biology-courses/microscopy-series/

## UC Berkeley Prof. Laura Waller
- https://www.youtube.com/watch?v=nlMqwWDLnfA&t=1540s

##  Exposure: A White-Box Photo Post-Processing Framework
#### ACM Transactions on Graphics (presented at SIGGRAPH 2018)
[Yuanming Hu](http://taichi.graphics/me/)<sup>1,2</sup>, [Hao He](https://github.com/hehaodele)<sup>1,2</sup>, Chenxi Xu<sup>1,3</sup>, [Baoyuan Wang](https://sites.google.com/site/zjuwby/)<sup>1</sup>, [Stephen Lin](https://www.microsoft.com/en-us/research/people/stevelin/)<sup>1</sup>

#### [[Paper](https://arxiv.org/abs/1709.09602)] [[PDF Slides](https://github.com/yuanming-hu/exposure/releases/download/slides/exposure-slides.pdf)] [[PDF Slides with notes](https://github.com/yuanming-hu/exposure/releases/download/slides/exposure-slides-with-notes.pdf)] [[SIGGRAPH 2018 Fast Forward](https://www.youtube.com/watch?v=JdTkKhm0LVU)]

## MIT Optics Class 
- Prof.George Barbastathis (https://www.youtube.com/watch?v=IYBYmOVmICg)

## VidLanKD: Improving Language Understanding via Video-Distilled Knowledge Transfer
* Arxiv: https://arxiv.org/abs/2107.02681
* https://github.com/zinengtang/VidLanKD
## attention-cnn
- [code](https://epfml.github.io/attention-cnn/)

#### Image Recognition 
* End-to-End Object Detection with Transformers (ECCV 2020)
    * [Original Paper Link](https://arxiv.org/abs/2005.12872) / [Paper Review Video](https://www.youtube.com/watch?v=hCWUTvVrG7E) / [Summary PDF](/lecture_notes/DETR.pdf) / Code Practice
    * 
* Searching for MobileNetV3 (ICCV 2019)
    * [Original Paper Link](https://arxiv.org/abs/1905.02244) / Paper Review Video / Summary PDF / Code Practice
* Deep Residual Learning for Image Recognition (CVPR 2016)
    * [Original Paper Link](https://arxiv.org/abs/1512.03385) / [Paper Review Video](https://www.youtube.com/watch?v=671BsKl8d0E) / [Summary PDF](/lecture_notes/ResNet.pdf) / [MNIST](/code_practices/ResNet18_MNIST_Train.ipynb) / [CIFAR-10](/code_practices/ResNet18_CIFAR10_Train.ipynb) / [ImageNet](/code_practices/Pretrained_ResNet18_ImageNet_Test.ipynb)
    
* Image Style Transfer Using Convolutional Neural Networks (CVPR 2016)
    * [Original Paper Link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) / Paper Review Video / Summary PDF / Code Practice
* Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (NIPS 2015)
    * [Original Paper Link](https://arxiv.org/abs/1506.01497) / Paper Review Video / Summary PDF / Code Practice

### ReLabel:
소설 프레임워크로 이미지넷 평가를 멀티 레이블 작업으로 돌릴 수 있습니다.
페이퍼: https://arxiv.org/pdf/2101.05022.pdf
Github: https://github.com/naver-ai/relabel_imagenet

## ML-HyperSim
Paper: https://arxiv.org/pdf/2011.02523.pdf
Codes: https://github.com/apple/ml-hypersim

## 딥페이스 드로잉: 스케치의 깊은 세대의 얼굴 이미지
종이: http://geometrylearning.com/paper/DeepFaceDrawing.pdf
비디오: https://www.youtube.com/watch?v=HSunooUTwKs

### Video Frame Interpolation 
* RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation
    * [paper link[(https://arxiv.org/pdf/2011.06294) 
https://github.com/hzwer/arXiv2020-RIFE

## MIT Researchers Propose Patch-Based Inference to Reduce the Memory Usage for Tiny Deep Learning
Quick Read: https://www.marktechpost.com/.../mit-researchers-propose.../
Paper: https://arxiv.org/abs/2110.15352
Project: https://mcunet.mit.edu/

## 시카고 대학과 텔아비브 대학의 연구진은 ‘텍스트2메쉬’를 소개했다: 텍스트 대상에 따라 3D 메쉬의 색과 기하학을 모두 바꾸는 소설 프레임워크
논문 요약: https://www.marktechpost.com/.../researchers-from-the.../
페이퍼: https://arxiv.org/pdf/2112.03221.pdf
GitHub: https://github.com/threedle/text2mesh
프로젝트 페이지: https://threedle.github.io/text2mesh/

## : A New Neural Network-Based Method To Build Animatable 3D Models From Videos
Quick Read: https://www.marktechpost.com/.../meta-ai-and-cmu.../
Paper: https://arxiv.org/pdf/2112.12761.pdf
Project: https://banmo-www.github.io/

## Stanford University/NVIDIA, via Sergio Valmorisco Sierra:
" Current state-of-the-art GANs have seen immense progress, but they commonly operate in 2D and do not explicitly model the underlying 3D scenes. Recent work on 3D-aware GANs has begun to tackle the problem of multi-view-consistent image synthesis and, to a lesser extent, extraction of 3D shapes without being supervised on geometry or multi-view image collections. However, the image quality and resolution of existing 3D GANs have lagged far behind those of 2D GANs. One of the primary reasons for this gap is the computational inefficiency of previously employed 3D generators and neural rendering architectures.
The authors of this paper introduce a novel generator architecture for unsupervised 3D representation learning from a collection of single-view 2D photographs that seeks to improve the computational efficiency of rendering while remaining true to 3D-grounded neural rendering.
For this purpose, the authors introduce an expressive hybrid explicit-implicit network architecture that, together with other design choices, synthesizes not only high-resolution multi-view-consistent images in real-time but also produces high-quality 3D geometry. By decoupling feature generation and neural rendering, their framework is able to leverage state-of-the-art 2D CNN generators, such as StyleGAN2, and inherit their efficiency and expressiveness."
 - Project: https://lnkd.in/dCV3t4qG
 - Code: https://lnkd.in/dRf8S-4n
 - Video: https://lnkd.in/d4hp2WAh
 - Paper: https://lnkd.in/datCy3HN
 - Authors: Eric R. Chan, Connor Lin, Matthew A. Chan, Koki Nagano, Boxiao (Leo) Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, Gordon Wetzstein

## Researchers at Meta and the University of Texas at Austin Propose ‘Detic’: A Method to Detect Twenty-Thousand Classes using Image-Level Supervision
Quick Read: https://www.marktechpost.com/.../researchers-at-meta-and.../
Paper: https://arxiv.org/pdf/2201.02605v2.pdf
Github: https://github.com/facebookresearch/Detic

## Apple ML Researchers Introduce ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene Understanding Using Mobile RGB-D Data
Quick Read: https://www.marktechpost.com/.../apple-ml-researchers.../
Paper: https://arxiv.org/pdf/2111.08897.pdf
Github: https://github.com/apple/ARKitScenes

## Researchers From China Propose A Pale-Shaped Self-Attention (PS-Attention) And A General Vision Transformer Backbone, Called Pale Transformer
Quick Read: https://www.marktechpost.com/.../researchers-from-china.../
Paper: https://arxiv.org/pdf/2112.14000v1.pdf
Github: https://github.com/BR-IDL/PaddleViT

## Meta AI and CMU Researchers Present ‘BANMo’: A New Neural Network-Based Method To Build Animatable 3D Models From Videos
Quick Read: https://www.marktechpost.com/.../meta-ai-and-cmu.../
Paper: https://arxiv.org/pdf/2112.12761.pdf
Project: https://banmo-www.github.io/

## PyTorch-StudioGAN 
깃허브 링크: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN

## UNETR’: A Novel Architecture for Semantic Segmentation of Brain Tumors Using Multi-Modal MRI Images
Quick Read: https://www.marktechpost.com/.../researchers-from-nvidia.../
Paper: https://arxiv.org/pdf/2201.01266v1.pdf
Github: https://github.com/.../research.../tree/master/SwinUNETR

## 비전 기반 제스처 제어 드론
젯슨나노2GB에 단안카메라로 제스쳐 모양에 따라 드론이 스탠드 후버링은 물론 좌/우로 제어가 되네요!
* 공헌자(저자): A FINDELAIR
* 데모 동영상: https://youtu.be/FZAUPmKiSXg 
* GitHub: https://github.com/ArthurFDLR/drone-gesture-control 
* 프로젝트 보고서: https://github.com/.../blob/main/.github/Project_Report.pdf 
* 데브토크: https://forums.developer.nvidia.com/.../vision.../187072 

## Google AI Introduces ‘StylEx’: A New Approach For A Visual Explanation Of Classifiers
Paper: https://arxiv.org/pdf/2104.13369.pdf
Project: https://explaining-in-style.github.io/
Github: https://github.com/google/explaining-in-style

## Meta AI Research Proposes ‘OMNIVORE’: 
A Single Vision (Computer Vision) Model For Many Different Visual Modalities
Quick Read: https://www.marktechpost.com/2022/01/30/meta-ai-research-proposes-omnivore-a-single-vision-computer-vision-model-for-many-different-visual-modalities/
Paper: https://arxiv.org/abs/2201.08377
Github: https://github.com/facebookresearch/omnivore

## SegFormer, a new semantic segmentation method has been proposed. 
You can read about it in the link below: https://arxiv.org/abs/2105.15203
I have prepared a python script which can be used to train the SegFormer model or alternatively you can download the model that I have trained and use it for the inference. The details for training and testing the model for semantic segmentation using SegFormer are available at my GitHub Repository. 
https://github.com/.../Sematic_Segmentation_With_SegFormer 
The sample inference from a test image is shown below:

##  Dynamic Shifting Network
자율주행에 대한 온전한 인식을 제공하기 위한 최신 컴퓨터 비전 연구에서 연구자들은 DSN,  LiDAR 기반 판옵틱 세그먼트의 과제를 다루고 있다
빠른 읽기: https://www.marktechpost.com/.../in-a-latest-computer.../
종이: https://arxiv.org/pdf/2203.07186v1.pdf
Github: https://github.com/hongfz16/DS-Net

## 3D Model Reconstruction
Images with a single camera can generate 3d model.
https://kotai2003-faces.streamlit.app/

## Face Pyramid Vision Transformer
- 33rd British Machine Vision Conference (BMVC) 2022, 21st - 24th November 2022, London, UK. 
- Project Page: https://khawar-islam.github.io/fpvt/
- Code: https: https://github.com/khawar-islam/FPVT_BMVC22

#### Highlights:
A novel Face Pyramid Vision Transformer (FPVT) is proposed to learn a discriminative multi-scale facial representations for face recognition and verification. In FPVT, Face Spatial Reduction Attention (FSRA) and Dimensionality Reduction (FDR) layers are employed to make the feature maps compact, thus reducing the computations. An Improved Patch Embedding (IPE) algorithm is proposed to exploit the benefits of CNNs in ViTs (e.g., shared weights, local context, and receptive fields) to model lower-level edges to higher-level semantic primitives. Within FPVT framework, a Convolutional Feed-Forward Network (CFFN) is proposed that extracts locality information to learn low level facial information. The proposed FPVT is evaluated on seven benchmark datasets and compared with ten existing state-of-the-art methods, including CNNs, pure ViTs, and Convolutional ViTs. Despite fewer parameters, FPVT has demonstrated excellent performance over the compared methods. I am greatly thankful to my coauthors Arif Mahmood and Zaigham Zaheer for their supervision and guidance throughout the project.

## DeepMind Introduces the Perception Test, a New Multimodal Benchmark Using Real-World Videos to Help Evaluate the Perception Capabilities of a Machine Learning Model
Paper: https://storage.googleapis.com/.../perception_test_report...
Github link: https://github.com/deepmind/perception_test


## Latest Computer Vision Research Proposes Lumos for Relighting Portrait Images via a Virtual Light Stage and Synthetic-to-Real Adaptation
Paper: https://arxiv.org/pdf/2209.10510.pdf
Demo: http://imaginaire.cc/Lumos/
Project: https://deepimagination.cc/Lumos/

## Google AI Introduces Frame Interpolation for Large Motion (FILM): A New Neural Network Architecture To Create High-Quality Slow-Motion Videos From Near-Duplicate Photos
Paper: https://arxiv.org/pdf/2202.04901.pdf
Github: https://github.com/google-research/frame-interpolation
Project: https://film-net.github.io/

## Latest Computer Vision Research at Nanyang Technological University Introduces VToonify Framework for Style Controllable High-Resolution Video Toonification
Paper: https://arxiv.org/pdf/2209.11224v2.pdf
Github: https://github.com/williamyang1991/vtoonify

## Salesforce AI Open-Sources ‘LAVIS,’ A Deep Learning Library For Language-Vision Research/Applications
Paper: https://arxiv.org/pdf/2209.09019.pdf
Github link: https://github.com/salesforce/LAVIS

## Researchers at Tencent Propose GFP-GAN that Leverages Rich and Diverse Priors Encapsulated in a Pretrained Face GAN for Blind Face Restoration
Paper: https://arxiv.org/pdf/2101.04061v2.pdf
Github link: https://github.com/TencentARC/GFPGAN

## Latest Computer Vision Research at Google and Boston University Proposes ‘DreamBooth,’ 
A Technique for Fine-Tuning a Text-to-Image Model with a very Limited Set of Images
Paper: https://arxiv.org/pdf/2208.12242.pdf?
Project: https://dreambooth.github.io/

## Researchers from McGill University and Microsoft Introduces Convolutional vision Transformer (CvT) that improves Vision Transformer (ViT) in Performance and Efficiency by Introducing Convolutions into ViT
Paper: https://openaccess.thecvf.com/.../Wu_CvT_Introducing...
GIthub: https://github.com/microsoft/CvT

## Microsoft Research Introduces a General-Purpose Multimodal Foundation Model ‘BEIT-3,’ that Achieves State-of-the-Art Transfer Performance on Both Vision and Vision Language Tasks
Paper: https://arxiv.org/pdf/2208.10442.pdf
Github: https://github.com/microsoft/unilm/tree/master/beit

## Apple Researchers Develop NeuMan
A Novel Computer Vision Framework that can Generate Neural Human Radiance Field from a Single Video
Paper: https://arxiv.org/pdf/2203.12575v1.pdf
Github: https://github.com/apple/ml-neuman

## Researchers from the Alibaba Group added their newly developed ‘YOLOX-PAI’ into EasyCV, which is an all-in-one Computer Vision Toolbox
Paper: https://arxiv.org/pdf/2208.13040v1.pdf
Github link: https://github.com/alibaba/EasyCV

## Deepmind Researchers Introduce ‘Transframer’: 
A General-Purpose AI Framework For Image Modelling And Computer Vision Tasks Based On Probabilistic Frame Prediction
Paper: https://arxiv.org/pdf/2203.09494.pdf

## Researchers at Apple Develop Texturify: A GAN-based Approach for Generating Textures on 3D Shape Surfaces
Paper: https://nihalsid.github.io/texturify/static/Texturify.pdf
Project: https://nihalsid.github.io/texturify/

## Latest Computer Vision Research At Microsoft Explains How This Proposed Method Adapts The Pretrained Language Image Models To Video Recognition
Paper: https://arxiv.org/pdf/2208.02816v1.pdf
Github: https://github.com/microsoft/VideoX/tree/master/X-CLIP

## Salesforce AI Propose A Novel Framework That Trains An Open Vocabulary Object Detector With Pseudo Bounding-Box Labels Generated From Large-Scale Image-Caption Pairs
Paper: https://arxiv.org/pdf/2111.09452.pdf
GItHub link: https://arxiv.org/pdf/2111.09452.pdf

## Researchers Present A Survey Report on Using 100+ Transformer-based Methods in Computer Vision for Different 3D Vision Tasks
Paper:  https://arxiv.org/pdf/2208.04309v1.pdf
Github link: https://github.com/lahoud/3d-vision-transformers

## Researchers Propose Easter2.0, a Novel Convolutional Neural Network CNN-Based Architecture for the Task of End-to-End Handwritten Text Line Recognition that Utilizes Only 1D Convolutions
Paper: https://arxiv.org/pdf/2205.14879v1.pdf
Github link: https://github.com/kartikgill/easter2

## Researchers at Meta AI Develop Multiface: A Dataset for Neural Face Rendering
Paper: https://arxiv.org/pdf/2207.11243v1.pdf
Github link: https://github.com/facebookresearch/multiface

## Research From China Propose a Novel Context-Aware Vision Transformer (CA-ViT) For Ghost-Free High Dynamic Range Imaging
They propose a novel vision transformer termed CA-ViT that can fully utilize both global and local picture context dependencies while outperforming its predecessors by a wide margin.
They introduce a unique HDR-Transformer that can reduce processing costs, ghosting artifacts, and recreating high-quality HDR photos. This is the first Transformer-based HDR de-ghosting framework to be developed. 
They undertake in-depth tests on three sample benchmark HDR datasets to compare HDR-performance Transformers to current state-of-the-art techniques.
Paper: https://arxiv.org/pdf/2208.05114v1.pdf
Github link: https://github.com/megvii-research/HDR-Transformer

## DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation (CVPR 2022) 
Abstract: Recently, GAN inversion methods combined with Contrastive Language-Image Pretraining (CLIP) enables zero-shot image manipulation guided by text prompts. However, their applications to diverse real images are still difficult due to the limited GAN inversion capability. Specifically, these approaches often have difficulties in reconstructing images with novel poses, views, and highly variable contents compared to the training data, altering object identity, or producing unwanted image artifacts. To mitigate these problems and enable faithful manipulation of real images, we propose a novel method, dubbed DiffusionCLIP, that performs text-driven image manipulation using diffusion models. Based on full inversion capability and high-quality image generation power of recent diffusion models, our method performs zero-shot image manipulation successfully even between unseen domains and takes another step towards general application by manipulating images from a widely varying ImageNet dataset. Furthermore, we propose a novel noise combination method that allows straightforward multi-attribute manipulation. Extensive experiments and human evaluation confirmed robust and superior manipulation performance of our methods compared to the existing baselines. 

Source: https://openaccess.thecvf.com/.../Kim_DiffusionCLIP_Text...
Slides: https://www.slideshare.net/.../diffusionclip-textguided...
Video: https://youtu.be/YVCtaXw6fw8
Code: https://github.com/gwang-kim/DiffusionCLIP.git

## NVIDIA AI Researchers Propose ‘MinVIS,’ 
A Minimal Video Instance Segmentation (VIS) Framework That Achieves SOTA Performance With Neither Video-Based Architectures Nor Training Procedures
Paper: https://arxiv.org/pdf/2208.02245v1.pdf
Github link: https://github.com/nvlabs/minvis

## Researchers from China Propose DAT: a Deformable Vision Transformer to Compute Self-Attention in a Data-Aware Fashion
Paper: https://openaccess.thecvf.com/.../Xia_Vision_Transformer...
Github: https://github.com/LeapLabTHU/DAT

## Researchers From CMU And Stanford Develop OBJECTFOLDER 2.0: A Multisensory Object Dataset For Sim2Real Transfer
Paper: https://arxiv.org/pdf/2204.02389.pdf
Github: https://github.com/rhgao/ObjectFolder
Project: https://ai.stanford.edu/~rhgao/objectfolder2.0/

## image classification on small-datasets in Pytorch
https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch

## Alibaba AI Research Team Introduces ‘DCT-Net’
A Novel Image Translation Architecture For Few-Shot Portrait Stylization
Paper: https://arxiv.org/pdf/2207.02426v1.pdf
Project: https://menyifang.github.io/projects/DCTNet/DCTNet.html
Github link: https://github.com/menyifang/dct-net

## NeurIPS2021 spotlight work PCAN-“Prototypical Cross-Attention Networks for Multiple Object Tracking and Segmentation”.
- PCAN uses test-time prototypes to memorize instance appearance and achieve impressive seg tracking accuracy on YT-VIS and BDD100K.
- Code: https://github.com/SysCV/pcan
- Paper: https://arxiv.org/abs/2106.11958

## yolo v7
YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors
공헌자(저자): Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
논문: https://arxiv.org/abs/2207.02696
GitHub: https://github.com/wongkinyiu/yolov7

초록: YOLOv7은 5FPS~160FPS 범위에서 속도와 정확도 모두에서 알려진 모든 객체 감지기를 능가하며 GPU V100에서 30FPS 이상의 알려진 모든 실시간 객체 감지기 중 가장 높은 정확도 56.8% AP를 가지고 있습니다. YOLOv7-E6 물체 감지기(56 FPS V100, 55.9% AP)는 변압기 기반 감지기인 SWIN-L Cascade-Mask R-CNN(9.2 FPS A100, 53.9% AP)보다 속도 509%, 정확도 2%, 컨볼루션 기반 검출기 ConvNeXt-XL Cascade-Mask R-CNN(8.6 FPS A100, 55.2% AP)은 속도 551%, AP 정확도 0.7% 향상 및 YOLOv7 성능 향상: YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, DETR, Deformable DETR, DINO-5scale-R50, ViT-Adapter-B 및 기타 여러 물체 감지기의 속도와 정확도. 또한 다른 데이터 세트나 사전 훈련된 가중치를 사용하지 않고 처음부터 MS COCO 데이터 세트에서만 YOLOv7을 훈련합니다.

## Diffusion Model
Diffusion 입문
https://lilianweng.github.io/.../2021-07-11-diffusion.../
https://youtu.be/d_x92vpIWFM
================================
Diffusion Models Beat GANs on Image Synthesis
확산 모델에 classifier를 추가해 다양성-품질 trade-off를 달성
Classifier-Free Diffusion Guidance
Classifier 없이 단일 확산 모델로 같은 목표 달성
Cascaded Diffusion Models for High Fidelity Image Generation
여러 해상도의 확산 모델이 포함된 계층적 cascading pipeline으로 이전보다 더 높은 해상도에서 고품질 샘플 생성
Pretraining is All You Need for Image-to-Image Translation (PITI)
사전 훈련된 확산 모델을 이용해 다양한 다운스트림 작업 입력 조건(e.g. semantic map + text)에서 Image-to-Image translation 수행

## akka-stream
다이나믹 배치를 구현하고 있고 충분히 작은 모델이라면 10000 ~ 20000  requests / sec 수준의 응답 성능을 낼 수 있습니다.
구현에 사용된 akka는 프레임워크가 아닌 고도의 동시성, 병렬성, 분산성을 가지고 있는 메세지 기반 어플레케이션 구축 툴킷으로 간주할 수 있습니다.
충분히 생산성이 있는 언어를 베이스로 하고 있기 때문에 직접 서빙 데몬에서 monolithic 한 구조로 비지니스 코드를 내재화 하는것도 가능합니다. 
예제의 코드량이 적고 추상화가 거의 없는 naive한 구현이기 때문에 동작에 관련한 거의 대부분의 요소를 블랙박스 없이 확인하고 동시에 환경 튜닝이 가능합니다.
실제 예제의 사용성은 웹과 상호 작용을 하는 어플리케이션 보다는 검색, 추천, 대화 시스템등 다수의 모델을 컨트롤하는 기반 플랫폼 시스템에 적합합니다.
https://github.com/go-noah/akka-dynamic-batch-serving/tree/main/akka-dynamic-batch-onnx-gpu-bert
https://github.com/go-noah/akka-dynamic-batch-serving/tree/main/akka-dynamic-batch-tensorflow-gpu-bert

## AI2’s PRIOR Team Introduces Unified-IO:
The First Neural Model To Execute Various AI Tasks Spanning Classical Computer Vision, Image Synthesis, Vision-and-Language, and Natural Language Processing NLP
Demo: https://unified-io.allenai.org/

## AI Researchers From China Introduce a New Vision GNN (ViG) Architecture to Extract Graph Level Feature for Visual Tasks
Paper: https://arxiv.org/pdf/2206.00272v1.pdf
Github: https://github.com/huawei-noah/Efficient-AI-Backbones

## Researchers at Stanford have developed an Artificial Intelligence (AI) model,
EG3D, that can generate random images of faces and other objects with high resolution together with underlying geometric structures
[Quick Read: https://www.marktechpost.com/2022/07/04/researchers-at-stanford-have-developed-an-artificial-intelligence-ai-model-eg3d-that-can-generate-random-images-of-faces-and-other-objects-with-high-resolution-together-with-underlying-geometric-s/?fbclid=IwAR3s59QXgJsrYG0uIiDTIIQl784LAUe48NrfJ6Vk6kTVVOjjHAzod7DRAEc
Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_Efficient_Geometry-Aware_3D_Generative_Adversarial_Networks_CVPR_2022_paper.pdf?fbclid=IwAR2oL0AvGr_0uBamWB67pHl_KNSAuhxN2VKpyzLcpGiIBVIyJiy7211j_8M
Github: https://github.com/NVlabs/eg3d

## Stanford and TRI AI Researchers Propose the Atemporal Probe (ATP), A New ML Model For Video-Language Analysis
Paper: https://arxiv.org/pdf/2206.01720.pdf
Project: https://stanfordvl.github.io/atp-revisit-video-lang/

## A New Technique to Train Diffusion Model in Latent Space Using Limited Computational Resources While Maintaining High-Resolution Quality
Paper: https://arxiv.org/pdf/2112.10752.pdf
Github: https://github.com/CompVis/latent-diffusion

## MPViT
Arxiv https://arxiv.org/abs/2112.11010
Code https://github.com/youngwanLEE/MPViT

## DN-DETR: Accelerate DETR Training by Introducing Query DeNoising
## DAB-DETR : Dynamic Anchor Boxes are Better Queries for DETR 

## Dynamic Gender Classification
code : https://github.com/CChenLi/Dynamic_Gender_Classification

## NVIDIA의 Efficient Geometry-aware 3D Generative Adversarial Networks(EG3D)
code: https://github.com/NVlabs/eg3d
paper: https://arxiv.org/abs/2112.07945
page: https://nvlabs.github.io/eg3d/
youtube: https://youtu.be/cXxEwI7QbKg

## Salesforce AI Research has proposed a new video-and-language representation learning framework called ALPRO. 
This framework can be used for pre-training models to achieve state-of-the-art performance on tasks such as video-text retrieval and question answering.
Paper: https://arxiv.org/pdf/2112.09583.pdf
Github: https://github.com/salesforce/alpro

## Warehouse Apparel Detection using YOLOv5 end to end project
Kindly Like and Share and subscribe to the YT channel !!
Project Code: https://github.com/Ashishkumar-hub/Warehouse-Apparel-Detection-using...

## Researchers From MIT and Cornell Develop STEGO 
(Self-Supervised Transformer With Energy-Based Graph Optimization): A Novel AI Framework That Distills Unsupervised Features Into High-Quality Discrete Semantic Labels
Paper: https://arxiv.org/pdf/2203.08414.pdf
Github: https://github.com/mhamilton723/STEGO

## UTokyo Researchers Introduce 
A Novel Synthetic Training Data Called Self-Blended Images (SBIs) To Detect Deepfakes
Paper: https://arxiv.org/pdf/2204.08376.pdf
Github: https://github.com/mapooon/SelfBlendedImages

## Meta AI Introduces ‘Make-A-Scene’: 
A Deep Generative Technique Based On An Autoregressive Transformer For Text-To-Image Synthesis With Human Priors
Paper Summary: https://www.marktechpost.com/.../meta-ai-introduces-make.../
Paper: https://arxiv.org/pdf/2203.13131v1.pdf

## Bytedance Researchers Propose CLIP-GEN: 
A New Self-Supervised Deep Learning Generative Approach Based On CLIP And VQ-GAN To Generate Reliable Samples From Text Prompts
Paper: https://arxiv.org/pdf/2203.00386v1.pdf

## Warehouse Apparel Detection using YOLOv5 end to end project
Kindly Like and Share and subscribe to the YT channel !!
Project Code: https://github.com/.../Warehouse-Apparel-Detection-using...

## Learning to Estimate Robust 3D Human Mesh from In-the-Wild Crowded Scenes / 3DCrowdNet
https://arxiv.org/abs/2104.07300
github: https://github.com/hongsukchoi/3DCrowdNet_RELEASE

## Google AI Researchers Propose SAVi++: 
An Object-Centric Video Model Trained To Predict Depth Signals From A Slot-Based Video Representation
Paper: https://arxiv.org/pdf/2206.07764.pdf
Project: https://slot-attention-video.github.io/savi++/

## Meta AI Research Proposes ‘OMNIVORE’: 
A Single Vision (Computer Vision) Model For Many Different Visual Modalities
Paper: https://arxiv.org/abs/2201.08377
Github: https://github.com/facebookresearch/omnivore

## 젯슨나노를 이용해서 녹색 이구아나의 외래 종의 실시간 탐지 및 모니터링
GitHub: https://github.com/.../Iguana-detection-on-Nvidia-Jetson...
블로그 링크: https://blogs.nvidia.com.tw/.../green-iguana-detection.../

