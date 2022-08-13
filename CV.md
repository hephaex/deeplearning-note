## Accelerating physics simulators for Robotics Reinforcement Learning - Erwin Coumans @ ICRA22 | 2/8
Pybullet을 만드신 분으로도 유명한 Erwin Coumans님의 ICRA 22 Tutorial:
강화학습을 로봇에 적용할 때 가장 많이 고민하는 시뮬레이터에 관한 좋은 발표입니다. 
Website : https://araffin.github.io/tools-for-robotic-rl-icra2022/
Slides : https://drive.google.com/.../19ImRxp8SfbTLtMDdFwYY.../view
Youtube : https://youtu.be/WOwLquiFbPE## Interested in multiple object tracking and segmentation and self-driving?

## Researchers Propose Easter2.0, a Novel Convolutional Neural Network CNN-Based Architecture for the Task of End-to-End Handwritten Text Line Recognition that Utilizes Only 1D Convolutions
Paper Summary: https://www.marktechpost.com/.../researchers-propose.../
Paper: https://arxiv.org/pdf/2205.14879v1.pdf
Github link: https://github.com/kartikgill/easter2

## Researchers at Meta AI Develop Multiface: A Dataset for Neural Face Rendering
Quick Read: https://www.marktechpost.com/.../researchers-at-meta-ai.../
Paper: https://arxiv.org/pdf/2207.11243v1.pdf
Github link: https://github.com/facebookresearch/multiface

## DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation (CVPR 2022) 
Abstract: Recently, GAN inversion methods combined with Contrastive Language-Image Pretraining (CLIP) enables zero-shot image manipulation guided by text prompts. However, their applications to diverse real images are still difficult due to the limited GAN inversion capability. Specifically, these approaches often have difficulties in reconstructing images with novel poses, views, and highly variable contents compared to the training data, altering object identity, or producing unwanted image artifacts. To mitigate these problems and enable faithful manipulation of real images, we propose a novel method, dubbed DiffusionCLIP, that performs text-driven image manipulation using diffusion models. Based on full inversion capability and high-quality image generation power of recent diffusion models, our method performs zero-shot image manipulation successfully even between unseen domains and takes another step towards general application by manipulating images from a widely varying ImageNet dataset. Furthermore, we propose a novel noise combination method that allows straightforward multi-attribute manipulation. Extensive experiments and human evaluation confirmed robust and superior manipulation performance of our methods compared to the existing baselines. 
Source: https://openaccess.thecvf.com/.../Kim_DiffusionCLIP_Text...
Slides: https://www.slideshare.net/.../diffusionclip-textguided...
Video: https://youtu.be/YVCtaXw6fw8
Code: https://github.com/gwang-kim/DiffusionCLIP.git

## Researchers from China Propose DAT: a Deformable Vision Transformer to Compute Self-Attention in a Data-Aware Fashion
Paper Summary: https://www.marktechpost.com/.../researchers-from-china.../
Paper: https://openaccess.thecvf.com/.../Xia_Vision_Transformer...
Github: https://github.com/LeapLabTHU/DAT

## Researchers From CMU And Stanford Develop OBJECTFOLDER 2.0: A Multisensory Object Dataset For Sim2Real Transfer
Quick Read: https://www.marktechpost.com/.../researchers-from-cmu.../
Paper: https://arxiv.org/pdf/2204.02389.pdf
Github: https://github.com/rhgao/ObjectFolder
Project: https://ai.stanford.edu/~rhgao/objectfolder2.0/

## image classification on small-datasets in Pytorch
https://github.com/Harry-KIT/Image-Classification-on-small-datasets-in-Pytorch


## Alibaba AI Research Team Introduces ‘DCT-Net’
A Novel Image Translation Architecture For Few-Shot Portrait Stylization
Paper Summary: https://www.marktechpost.com/.../alibaba-ai-research.../
Paper: https://arxiv.org/pdf/2207.02426v1.pdf
Project: https://menyifang.github.io/projects/DCTNet/DCTNet.html
Github link: https://github.com/menyifang/dct-net

## NeurIPS2021 spotlight work PCAN-“Prototypical Cross-Attention Networks for Multiple Object Tracking and Segmentation”.
- PCAN uses test-time prototypes to memorize instance appearance and achieve impressive seg tracking accuracy on YT-VIS and BDD100K.
- Project website: https://vis.xyz/pub/pcan/
- Code: https://github.com/SysCV/pcan
- Paper: https://arxiv.org/abs/2106.11958

## yolo v7
YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors
공헌자(저자): Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
논문: https://arxiv.org/abs/2207.02696
GitHub: https://github.com/wongkinyiu/yolov7

초록: YOLOv7은 5FPS~160FPS 범위에서 속도와 정확도 모두에서 알려진 모든 객체 감지기를 능가하며 GPU V100에서 30FPS 이상의 알려진 모든 실시간 객체 감지기 중 가장 높은 정확도 56.8% AP를 가지고 있습니다. YOLOv7-E6 물체 감지기(56 FPS V100, 55.9% AP)는 변압기 기반 감지기인 SWIN-L Cascade-Mask R-CNN(9.2 FPS A100, 53.9% AP)보다 속도 509%, 정확도 2%, 컨볼루션 기반 검출기 ConvNeXt-XL Cascade-Mask R-CNN(8.6 FPS A100, 55.2% AP)은 속도 551%, AP 정확도 0.7% 향상 및 YOLOv7 성능 향상: YOLOR, YOLOX, Scaled-YOLOv4, YOLOv5, DETR, Deformable DETR, DINO-5scale-R50, ViT-Adapter-B 및 기타 여러 물체 감지기의 속도와 정확도. 또한 다른 데이터 세트나 사전 훈련된 가중치를 사용하지 않고 처음부터 MS COCO 데이터 세트에서만 YOLOv7을 훈련합니다.

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
Quick Read: https://www.marktechpost.com/.../ai2s-prior-team.../
Demo: https://unified-io.allenai.org/

## AI Researchers From China Introduce a New Vision GNN (ViG) Architecture to Extract Graph Level Feature for Visual Tasks
Paper Summary: https://www.marktechpost.com/.../ai-researchers-from.../
Paper: https://arxiv.org/pdf/2206.00272v1.pdf
Github: https://github.com/huawei-noah/Efficient-AI-Backbones

## Researchers at Stanford have developed an Artificial Intelligence (AI) model,
EG3D, that can generate random images of faces and other objects with high resolution together with underlying geometric structures
[Quick Read: https://www.marktechpost.com/2022/07/04/researchers-at-stanford-have-developed-an-artificial-intelligence-ai-model-eg3d-that-can-generate-random-images-of-faces-and-other-objects-with-high-resolution-together-with-underlying-geometric-s/?fbclid=IwAR3s59QXgJsrYG0uIiDTIIQl784LAUe48NrfJ6Vk6kTVVOjjHAzod7DRAEc
Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_Efficient_Geometry-Aware_3D_Generative_Adversarial_Networks_CVPR_2022_paper.pdf?fbclid=IwAR2oL0AvGr_0uBamWB67pHl_KNSAuhxN2VKpyzLcpGiIBVIyJiy7211j_8M
Github: https://github.com/NVlabs/eg3d

## Stanford and TRI AI Researchers Propose the Atemporal Probe (ATP), A New ML Model For Video-Language Analysis
Quick Read: https://www.marktechpost.com/.../stanford-and-tri-ai.../
Paper: https://arxiv.org/pdf/2206.01720.pdf
Project: https://stanfordvl.github.io/atp-revisit-video-lang/

## A New Technique to Train Diffusion Model in Latent Space Using Limited Computational Resources While Maintaining High-Resolution Quality
Paper Summary: [https://www.marktechpost.com/.../a-new-technique-to.../
](https://www.marktechpost.com/2022/06/28/a-new-technique-to-train-diffusion-model-in-latent-space-using-limited-computational-resources-while-maintaining-high-resolution-quality/?fbclid=IwAR1n777ssnw4C5oQ-TR8OcWug4jwXZ3uK-3LnCuFME2IgTi6VkqhoALBD_Y)
Paper: https://arxiv.org/pdf/2112.10752.pdf
Github: https://github.com/CompVis/latent-diffusion

## MPViT
Arxiv👉 https://arxiv.org/abs/2112.11010
Code👉 https://github.com/youngwanLEE/MPViT

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
Quick Read: https://www.marktechpost.com/.../salesforce-ai-research.../
Paper: https://arxiv.org/pdf/2112.09583.pdf
Github: https://github.com/salesforce/alpro

## Warehouse Apparel Detection using YOLOv5 end to end project
Kindly Like and Share and subscribe to the YT channel !!
Project Code: https://github.com/Ashishkumar-hub/Warehouse-Apparel-Detection-using...

## Researchers From MIT and Cornell Develop STEGO 
(Self-Supervised Transformer With Energy-Based Graph Optimization): A Novel AI Framework That Distills Unsupervised Features Into High-Quality Discrete Semantic Labels
Quick Read: https://www.marktechpost.com/.../researchers-from-mit.../
Paper: https://arxiv.org/pdf/2203.08414.pdf
Github: https://github.com/mhamilton723/STEGO

## UTokyo Researchers Introduce 
A Novel Synthetic Training Data Called Self-Blended Images (SBIs) To Detect Deepfakes
Quick Read: https://www.marktechpost.com/.../utokyo-researchers.../
Paper: https://arxiv.org/pdf/2204.08376.pdf
Github: https://github.com/mapooon/SelfBlendedImages

## Meta AI Introduces ‘Make-A-Scene’: 
A Deep Generative Technique Based On An Autoregressive Transformer For Text-To-Image Synthesis With Human Priors
Paper Summary: https://www.marktechpost.com/.../meta-ai-introduces-make.../
Paper: https://arxiv.org/pdf/2203.13131v1.pdf

## Bytedance Researchers Propose CLIP-GEN: 
A New Self-Supervised Deep Learning Generative Approach Based On CLIP And VQ-GAN To Generate Reliable Samples From Text Prompts
Quick Read: https://www.marktechpost.com/.../bytedance-researchers.../
Paper: https://arxiv.org/pdf/2203.00386v1.pdf

## Warehouse Apparel Detection using YOLOv5 end to end project
Kindly Like and Share and subscribe to the YT channel !!
Project Code: https://github.com/.../Warehouse-Apparel-Detection-using...

## Learning to Estimate Robust 3D Human Mesh from In-the-Wild Crowded Scenes / 3DCrowdNet
https://arxiv.org/abs/2104.07300
github: https://github.com/hongsukchoi/3DCrowdNet_RELEASE

## Google AI Researchers Propose SAVi++: 
An Object-Centric Video Model Trained To Predict Depth Signals From A Slot-Based Video Representation
Quick Read: https://www.marktechpost.com/.../google-ai-researchers.../
Paper: https://arxiv.org/pdf/2206.07764.pdf
Project: https://slot-attention-video.github.io/savi++/

## Meta AI Research Proposes ‘OMNIVORE’: 
A Single Vision (Computer Vision) Model For Many Different Visual Modalities
Quick Read: https://www.marktechpost.com/2022/01/30/meta-ai-research-proposes-omnivore-a-single-vision-computer-vision-model-for-many-different-visual-modalities/
Paper: https://arxiv.org/abs/2201.08377
Github: https://github.com/facebookresearch/omnivore

## 젯슨나노를 이용해서 녹색 이구아나의 외래 종의 실시간 탐지 및 모니터링
공헌자(저자): NVIDIA(타이완)
GitHub: https://github.com/.../Iguana-detection-on-Nvidia-Jetson...
블로그 링크: https://blogs.nvidia.com.tw/.../green-iguana-detection.../
