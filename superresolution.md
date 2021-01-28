
### DL based approach

Note this table is referenced from [here](https://github.com/LoSealL/VideoSuperResolution/blob/master/README.md#network-list-and-reference-updating)

### 2017

| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DRRN                   | [CVPR17](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf) | [Caffe](https://github.com/tyshiwo/DRRN_CVPR17), [PyTorch](https://github.com/jt827859032/DRRN-pytorch) | Recurrent                                                    |
| LapSRN                 | [CVPR17](http://vllab.ucmerced.edu/wlai24/LapSRN/)           | [Matlab](https://github.com/phoenix104104/LapSRN)            | Huber loss                                                   |
| IRCNN                  | [CVPR17](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_Deep_CNN_CVPR_2017_paper.pdf) | [Matlab](https://github.com/cszn/IRCNN)                      |                                                              |
| EDSR                   | [CVPR17](https://arxiv.org/abs/1707.02921)                   | [PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)       | NTIRE17 Champion                                             |
| BTSRN                  | [CVPR17](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Fan_Balanced_Two-Stage_Residual_CVPR_2017_paper.pdf) | -                                                            | NTIRE17                                                      |
| SelNet                 | [CVPR17](https://ieeexplore.ieee.org/document/8014887)       | -                                                            | NTIRE17                                                      |
| TLSR                   | [CVPR17](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Xu_Fast_and_Accurate_CVPR_2017_paper.pdf) | -                                                            | NTIRE17                                                      |
| SRGAN                  | [CVPR17](https://arxiv.org/abs/1609.04802)                   | [Tensorflow](https://github.com/tensorlayer/srgan)           | 1st proposed GAN                                             |
| VESPCN                 | [CVPR17](https://arxiv.org/abs/1611.05250)                   | -                                                            | **VideoSR**                                                  |
| MemNet                 | [ICCV17](https://arxiv.org/abs/1708.02209)                   | [Caffe](https://github.com/tyshiwo/MemNet)                   |                                                              |
| SRDenseNet             | [ICCV17](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf) | -, [PyTorch](https://github.com/wxywhu/SRDenseNet-pytorch)   | Dense                                                        |
| SPMC                   | [ICCV17](https://arxiv.org/abs/1704.02738)                   | [Tensorflow](https://github.com/jiangsutx/SPMC_VideoSR)      | **VideoSR**                                                  |
| EnhanceNet             | [ICCV17](https://arxiv.org/abs/1612.07919)                   | [TensorFlow](https://github.com/msmsajjadi/EnhanceNet-Code)  | Perceptual Loss                                              |
| PRSR                   | [ICCV17](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dahl_Pixel_Recursive_Super_ICCV_2017_paper.pdf) | [TensorFlow](https://github.com/nilboy/pixel-recursive-super-resolution) | an extension of PixelCNN                                     |
| AffGAN                 | [ICLR17](https://arxiv.org/pdf/1610.04490.pdf)               | -                                                            |                                                              |

### 2018
| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MS-LapSRN              | [TPAMI18](https://ieeexplore.ieee.org/document/8434354)      | [Matlab](https://github.com/phoenix104104/LapSRN)            | Fast LapSRN                                                  |
| DCSCN                  | [arXiv](https://arxiv.org/abs/1707.05425)                    | [Tensorflow](https://github.com/jiny2001/dcscn-super-resolution) |                                                              |
| IDN                    | [CVPR18](https://arxiv.org/abs/1803.09454)                   | [Caffe](https://github.com/Zheng222/IDN-Caffe)               | Fast                                                         |
| DSRN                   | [CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Han_Image_Super-Resolution_via_CVPR_2018_paper.pdf) | [TensorFlow](https://github.com/WeiHan3/dsrn/tree/db21d57dfab57de3608f0372e749c6488b6b305d) | Dual state，Recurrent                                        |
| RDN                    | [CVPR18](https://arxiv.org/abs/1802.08797)                   | [Torch](https://github.com/yulunzhang/RDN)                   | Deep, BI-BD-DN                                               |
| SRMD                   | [CVPR18](https://arxiv.org/abs/1712.06116)                   | [Matlab](https://github.com/cszn/SRMD)                       | Denoise/Deblur/SR                                            |
| xUnit                  | [CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kligvasser_xUnit_Learning_a_CVPR_2018_paper.pdf) | [PyTorch](https://github.com/kligvasser/xUnit)               | Spatial Activation Function                                  |
| DBPN                   | [CVPR18](https://arxiv.org/abs/1803.02735)                   | [PyTorch](https://github.com/alterzero/DBPN-Pytorch)         | NTIRE18 Champion                                             |
| WDSR                   | [CVPR18](https://arxiv.org/abs/1808.08718)                   | [PyTorch](https://github.com/JiahuiYu/wdsr_ntire2018)，[TensorFlow](https://github.com/ychfan/tf_estimator_barebone/blob/master/docs/super_resolution.md) | NTIRE18 Champion                                             |
| ProSRN                 | [CVPR18](https://arxiv.org/abs/1804.02900)                   | [PyTorch](https://github.com/fperazzi/proSR)                 | NTIRE18                                                      |
| ZSSR                   | [CVPR18](http://www.wisdom.weizmann.ac.il/~vision/zssr/)     | [Tensorflow](https://github.com/assafshocher/ZSSR)           | Zero-shot                                                    |
| FRVSR                  | [CVPR18](https://arxiv.org/abs/1801.04590)                   | [PDF](https://github.com/msmsajjadi/FRVSR)                   | **VideoSR**                                                  |
| DUF                    | [CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf) | [Tensorflow](https://github.com/yhjo09/VSR-DUF)              | **VideoSR**                                                  |
| TDAN                   | [arXiv](https://arxiv.org/pdf/1812.02898.pdf)                | -                                                            | **VideoSR**，Deformable Align                                |
| SFTGAN                 | [CVPR18](https://arxiv.org/abs/1804.02815)                   | [PyTorch](https://github.com/xinntao/SFTGAN)                 |                                                              |
| CARN                   | [ECCV18](https://arxiv.org/abs/1803.08664)                   | [PyTorch](https://github.com/nmhkahn/CARN-pytorch)           | Lightweight                                                  |
| RCAN                   | [ECCV18](https://arxiv.org/abs/1807.02758)                   | [PyTorch](https://github.com/yulunzhang/RCAN)                | Deep, BI-BD-DN                                               |
| MSRN                   | [ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf) | [PyTorch](https://github.com/MIVRC/MSRN-PyTorch)             |                                                              |
| SRFeat                 | [ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf) | [Tensorflow](https://github.com/HyeongseokSon1/SRFeat)       | GAN                                                          |
| TSRN                   | [ECCV18](https://arxiv.org/pdf/1808.00043.pdf)               | [Pytorch](https://github.com/waleedgondal/Texture-based-Super-Resolution-Network) |                                                              |
| ESRGAN                 | [ECCV18](https://arxiv.org/abs/1809.00219)                   | [PyTorch](https://github.com/xinntao/ESRGAN)                 | PRIM18 region 3 Champion                                     |
| EPSR                   | [ECCV18](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vasu_Analyzing_Perception-Distortion_Tradeoff_using_Enhanced_Perceptual_Super-resolution_Network_ECCVW_2018_paper.pdf) | [PyTorch](https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw) | PRIM18 region 1 Champion                                     |
| PESR                   | [ECCV18](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vu_Perception-Enhanced_Image_Super-Resolution_via_Relativistic_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) | [PyTorch](https://github.com/thangvubk/PESR)                 | ECCV18 workshop                                              |
| FEQE                   | [ECCV18](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Vu_Fast_and_Efficient_Image_Quality_Enhancement_via_Desubpixel_Convolutional_Neural_ECCVW_2018_paper.pdf) | [Tensorflow](https://github.com/thangvubk/FEQE)              | Fast                                                         |
| NLRN                   | [NIPS18](https://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration.pdf) | [Tensorflow](https://github.com/Ding-Liu/NLRN)               | Non-local, Recurrent                                         |
| SRCliqueNet            | [NIPS18](https://arxiv.org/abs/1809.04508)                   | -                                                            | Wavelet                                                      |
| CBDNet                 | [arXiv](https://arxiv.org/abs/1807.04686)                    | [Matlab](https://github.com/GuoShi28/CBDNet)                 | Blind-denoise                                                |
| TecoGAN                | [arXiv](http://arxiv.org/abs/1811.09393)                     | [Tensorflow](https://github.com/thunil/TecoGAN)              | **VideoSR** GAN       

### 2019
| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| RBPN                   | [CVPR19](https://arxiv.org/abs/1903.10128)                   | [PyTorch](https://github.com/alterzero/RBPN-PyTorch)         | **VideoSR**                                                  |
| SRFBN                  | [CVPR19](https://arxiv.org/abs/1903.09814)                   | [PyTorch](https://github.com/Paper99/SRFBN_CVPR19)           | Feedback                                                     |
| AdaFM                  | [CVPR19](https://arxiv.org/pdf/1904.08118.pdf)               | [PyTorch](https://github.com/hejingwenhejingwen/AdaFM)       | Adaptive Feature Modification Layers                         |
| MoreMNAS               | [arXiv](https://arxiv.org/pdf/1901.01074.pdf)                | -                                                            | Lightweight，NAS                                             |
| FALSR                  | [arXiv](https://arxiv.org/pdf/1901.07261.pdf)                | [TensorFlow](https://ieeexplore.ieee.org/document/8434354)   | Lightweight，NAS                                             |
| Meta-SR                | [CVPR19](https://arxiv.org/pdf/1903.00875.pdf)               | [PyTorch](https://github.com/XuecaiHu/Meta-SR-Pytorch)       | Arbitrary Magnification                                      |
| AWSRN                  | [arXiv](https://arxiv.org/abs/1904.02358)                    | [PyTorch](https://github.com/ChaofWang/AWSRN)                | Lightweight                                                  |
| OISR                   | [CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/He_ODE-Inspired_Network_Design_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf) | [PyTorch](https://github.com/HolmesShuan/OISR-PyTorch)       | ODE-inspired Network                                         |
| DPSR                   | [CVPR19](https://arxiv.org/pdf/1903.12529.pdf)               | [PyTorch](https://github.com/cszn/DPSR)                      |                                                              |
| DNI                    | [CVPR19](https://arxiv.org/pdf/1811.10515.pdf)               | [PyTorch](https://github.com/xinntao/DNI)                    |                                                              |
| MAANet                 | [arXiv](https://arxiv.org/abs/1904.06252)                    |                                                              | Multi-view Aware Attention                                   |
| RNAN                   | [ICLR19](https://openreview.net/pdf?id=HkeGhoA5FX)           | [PyTorch](https://github.com/yulunzhang/RNAN)                | Residual Non-local Attention                                 |
| FSTRN                  | [CVPR19](https://arxiv.org/pdf/1904.02870.pdf)               | -                                                            | **VideoSR**, fast spatio-temporal residual block             |
| MsDNN                  | [arXiv](https://arxiv.org/pdf/1904.10698.pdf)                | [TensorFlow](https://github.com/shangqigao/gsq-image-SR)     | NTIRE19  real SR  21th place                                 |
| SAN                    | [CVPR19](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-SAN.pdf) | [Pytorch](https://github.com/daitao/SAN)                     | Second-order Attention,cvpr19 oral                           |
| EDVR                   | [CVPRW19](https://arxiv.org/pdf/1905.02716.pdf)              | [Pytorch](https://github.com/xinntao/EDVR)                   | **Video**, NTIRE19 video restoration and enhancement champions |
| Ensemble for VSR       | [CVPRW19](https://arxiv.org/pdf/1905.02462.pdf)              | -                                                            | **VideoSR**, NTIRE19 video SR 2nd place                      |
| TENet                  | [arXiv](https://arxiv.org/pdf/1905.02538.pdf)                | [Pytorch](https://github.com/guochengqian/TENet)             | a Joint Solution for Demosaicking, Denoising and Super-Resolution |
| MCAN                   | [arXiv](https://arxiv.org/pdf/1903.07949.pdf)                | [Pytorch](https://github.com/macn3388/MCAN)                  | Matrix-in-matrix CAN, Lightweight                            |
| IKC&SFTMD              | [CVPR19](https://arxiv.org/pdf/1904.03377.pdf)               | -                                                            | Blind Super-Resolution                                       |
| SRNTT                  | [CVPR19](https://arxiv.org/pdf/1903.00834.pdf)               | [TensorFlow](https://github.com/ZZUTK/SRNTT)                 | Neural Texture Transfer                                      |
| RawSR                  | [CVPR19](https://arxiv.org/pdf/1905.12156.pdf)               | [TensorFlow](https://drive.google.com/file/d/1yvCceNAgt4UsxZXahPFBkuL1JXyfgr8B/view) | Real Scene Super-Resolution, Raw Images                      |
| resLF                  | [CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Residual_Networks_for_Light_Field_Image_Super-Resolution_CVPR_2019_paper.pdf) |                                                              | Light field                                                  |
| CameraSR               | [CVPR19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Camera_Lens_Super-Resolution_CVPR_2019_paper.pdf) |                                                              | realistic image SR                                           |
| ORDSR                  | [TIP](https://arxiv.org/pdf/1904.10082.pdf)                  | [model](https://github.com/tT0NG/ORDSR)                      | DCT domain SR                                                |
| U-Net                  | [CVPRW19](https://arxiv.org/pdf/1906.04809.pdf)              |                                                              | NTIRE19  real SR  2nd place, U-Net,MixUp,Synthesis           |
| DRLN                   | [arxiv](https://arxiv.org/pdf/1906.12021.pdf)                |                                                              | Densely Residual Laplacian Super-Resolution                  |
| EDRN                   | [CVPRW19](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Cheng_Encoder-Decoder_Residual_Network_for_Real_Super-Resolution_CVPRW_2019_paper.pdf) | [Pytorch](https://github.com/yyknight/NTIRE2019_EDRN)        | NTIRE19  real SR  9th places                                 |
| FC2N                   | [arXiv](https://arxiv.org/pdf/1907.03221.pdf)                |                                                              | Fully Channel-Concatenated                                   |
| GMFN                   | [BMVC2019](https://arxiv.org/pdf/1907.04253.pdf)             | [Pytorch](https://github.com/liqilei/GMFN)                   | Gated Multiple Feedback                                      |
| CNN&TV-TV Minimization | [BMVC2019](https://arxiv.org/pdf/1907.05380.pdf)             |                                                              | TV-TV Minimization                                           |
| HRAN                   | [arXiv](https://arxiv.org/pdf/1907.05514.pdf)                |                                                              | Hybrid Residual Attention Network                            |
| PPON                   | [arXiv](https://arxiv.org/pdf/1907.10399.pdf)                | [code](https://github.com/Zheng222/PPON)                     | Progressive Perception-Oriented Network                      |
| SROBB                  | [ICCV19](https://arxiv.org/pdf/1908.07222.pdf)               |                                                              | Targeted Perceptual Loss                                     |
| RankSRGAN              | [ICCV19](https://arxiv.org/pdf/1908.06382.pdf)               | [PyTorch](https://github.com/WenlongZhang0724/RankSRGAN)     | oral, rank-content loss                                      |
| edge-informed          | [ICCVW19](https://arxiv.org/pdf/1909.05305.pdf)              | [PyTorch](https://github.com/knazeri/edge-informed-sisr)     | Edge-Informed Single Image Super-Resolution                  |
| s-LWSR                 | [arxiv](https://arxiv.org/pdf/1909.10774.pdf)                |                                                              | Lightweight                                                  |
| DNLN                   | [arxiv](https://arxiv.org/pdf/1909.10692.pdf)                |                                                              | **Video SR** Deformable Non-local Network                    |
| MGAN                   | [arxiv](https://arxiv.org/pdf/1909.11937.pdf)                |                                                              | Multi-grained Attention Networks                             |
| IMDN                   | [ACM MM 2019](https://arxiv.org/pdf/1909.11856.pdf)          | [PyTorch](https://github.com/Zheng222/IMDN)                  | AIM19  Champion                                              |
| ESRN                   | [arxiv](https://arxiv.org/pdf/1909.11409.pdf)                |                                                              | NAS                                                          |
| PFNL                   | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yi_Progressive_Fusion_Video_Super-Resolution_Network_via_Exploiting_Non-Local_Spatio-Temporal_Correlations_ICCV_2019_paper.pdf) | [Tensorflow](https://github.com/psychopa4/PFNL)              | **VideoSR** oral,Non-Local Spatio-Temporal Correlations      |
| EBRN                   | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qiu_Embedded_Block_Residual_Network_A_Recursive_Restoration_Model_for_Single-Image_ICCV_2019_paper.pdf) |  [Tensorflow](https://github.com/alilajevardi/Embedded-Block-Residual-Network)                      | Embedded Block Residual Network                              |
| Deep SR-ITM            | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kim_Deep_SR-ITM_Joint_Learning_of_Super-Resolution_and_Inverse_Tone-Mapping_for_ICCV_2019_paper.pdf) | [matlab](https://github.com/sooyekim/Deep-SR-ITM)            | SDR to HDR, 4K SR                                            |
| feature SR             | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Noh_Better_to_Follow_Follow_to_Be_Better_Towards_Precise_Supervision_ICCV_2019_paper.pdf) |                                                              | Super-Resolution for Small Object Detection                  |
| STFAN                  | [ICCV19](https://arxiv.org/pdf/1904.12257.pdf)               | [PyTorch](https://github.com/sczhou/STFAN)                   | **Video Deblurring**                                         |
| KMSR                   | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Kernel_Modeling_Super-Resolution_on_Real_Low-Resolution_Images_ICCV_2019_paper.pdf) | [PyTorch](https://github.com/IVRL/Kernel-Modeling-Super-Resolution) | GAN for blur-kernel estimation                               |
| CFSNet                 | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_CFSNet_Toward_a_Controllable_Feature_Space_for_Image_Restoration_ICCV_2019_paper.pdf) | [PyTorch](https://github.com/qibao77/CFSNet)                 | Controllable Feature                                         |
| FSRnet                 | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gu_Fast_Image_Restoration_With_Multi-Bin_Trainable_Linear_Units_ICCV_2019_paper.pdf) |                                                              | Multi-bin Trainable Linear Units                             |
| SAM+VAM                | [ICCVW19](https://arxiv.org/pdf/1911.08711.pdf)              |                                                              |                                                              |
| SinGAN                 | [ICCV19](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) |    [PyTorch](https://github.com/tamarott/SinGAN)              | bestpaper,  train from single image   |

### 2020
| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| FISR                   | [AAAI 20](https://arxiv.org/pdf/1912.07213.pdf)            | [TensorFlow](https://github.com/JihyongOh/FISR)              | **Video joint VFI-SR method**,Multi-scale Temporal Loss      |
| ADCSR                  | [arxiv](https://arxiv.org/pdf/1912.08002.pdf)                |                                                              |                                                              |
| SCN                    | [AAAI 20](https://arxiv.org/pdf/1912.09028.pdf)            |                                                              | Scale-wise Convolution                                       |
| LSRGAN                 | [arxiv](https://arxiv.org/pdf/2001.08126.pdf)            |                                                              | Latent Space Regularization for srgan                                       |
| Zooming Slow-Mo        | [CVPR 20](https://arxiv.org/pdf/2002.11616.pdf)            | [PyTorch](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)                                                             | joint VFI and SR，one-stage，  deformable ConvLSTM                     |
| MZSR                   | [CVPR 20](https://arxiv.org/pdf/2002.12213.pdf)            |                                                              | Meta-Transfer Learning, Zero-Shot                     |
| VESR-Net               | [arxiv](https://arxiv.org/pdf/2003.02115.pdf)            |                                                              | Youku Video Enhancement and Super-Resolution Challenge Champion                    |
| blindvsr               | [arxiv](https://arxiv.org/pdf/2003.04716.pdf)            |   [PyTorch](https://github.com/jspan/blindvsr)                                                           | Motion blur estimation                    |
| HNAS-SR                | [arxiv](https://arxiv.org/pdf/2003.04619.pdf)            |   [PyTorch](https://github.com/guoyongcs/HNAS-SR)                                                           | Hierarchical Neural Architecture Search, Lightweight   |
| DRN                    | [CVPR 20](https://arxiv.org/pdf/2003.07018.pdf)            | [PyTorch](https://github.com/guoyongcs/DRN)                                                             | Dual Regression, SISR STOA                     |
| SFM                    | [arxiv](https://arxiv.org/pdf/2003.07119.pdf)            |   [PyTorch](https://github.com/sfm-sr-denoising/sfm)                                                           | Stochastic Frequency Masking, Improve method                     |
| EventSR                | [CVPR 20](https://arxiv.org/pdf/2003.07640.pdf)            |                                                             | split three phases                     |
| USRNet                 | [CVPR 20](https://arxiv.org/pdf/2003.10428.pdf)            | [PyTorch](https://github.com/cszn/USRNet)                                                             |                    |
| PULSE                  | [CVPR 20](https://arxiv.org/pdf/2003.03808.pdf)            |  [PyTorch](https://github.com/krantirk/Self-Supervised-photo)      | Self-Supervised                     |
| SPSR                   | [CVPR 20](https://arxiv.org/pdf/2003.13081.pdf)            | [Code](https://github.com/Maclory/SPSR)                                                             |  Gradient Guidance, GAN                  |
| DASR                   | [arxiv](https://arxiv.org/pdf/2004.01178.pdf)            | [Code](https://github.com/ShuhangGu/DASR)                                                             | Real-World Image Super-Resolution, Unsupervised SuperResolution, Domain Adaptation.                  |
| STVUN                  | [arxiv](https://arxiv.org/pdf/2004.02432.pdf)            | [PyTorch](https://github.com/JaeYeonKang/STVUN-Pytorch)                                                             | Video Super-Resolution, Video Frame Interpolation, Joint space-time upsampling                  |
| AdaDSR                 | [arxiv](https://arxiv.org/pdf/2004.03915.pdf)            | [PyTorch](https://github.com/csmliu/AdaDSR)                                                             | Adaptive Inference                  |
| Scale-Arbitrary SR     | [arxiv](https://arxiv.org/pdf/2004.03791.pdf)            | [Code](https://github.com/LongguangWang/Scale-Arbitrary-SR)                                                             | Scale-Arbitrary Super-Resolution, Knowledge Transfer                  |
| DeepSEE                | [arxiv](https://arxiv.org/pdf/2004.04433.pdf)            | [Code](https://mcbuehler.github.io/DeepSEE/)                                                             | Extreme super-resolution,32× magnification                  |
| CutBlur                | [CVPR 20](https://arxiv.org/pdf/2004.00448.pdf)            | [PyTorch](https://github.com/clovaai/cutblur/blob/master/main.py)             | SR Data Augmentation                  |
| UDVD                   | [CVPR 20](https://arxiv.org/pdf/2004.06965.pdf)            |   | Unified Dynamic Convolutional，SISR and denoise           |
|DIN                     | [arxiv](https://arxiv.org/pdf/2010.15689.pdf)  |[Code](https://github.com/lifengshiwo/DIN) | asymmetric co-attention, deep interleaved network  |
| PANet                  | [arxiv](https://arxiv.org/pdf/2004.13824.pdf)   |[PyTorch](https://github.com/SHI-Labs/Pyramid-Attention-Networks)   | Pyramid Attention        |
| SRResCGAN              | [arxiv](https://arxiv.org/pdf/2005.00953.pdf)   |[PyTorch](https://github.com/RaoUmer/SRResCGAN)   |         |
| ISRN                   | [arxiv](https://arxiv.org/pdf/2005.09964.pdf)   |   | iterative optimization, feature normalization.     |
| RFB-ESRGAN             | [CVPR 20](https://arxiv.org/pdf/2005.12597.pdf)   |   | NTIRE 2020 Perceptual Extreme Super-Resolution Challenge winner    |
| PHYSICS_SR             | [AAAI 20](https://arxiv.org/pdf/1908.06444.pdf)            | [PyTorch](https://github.com/jspan/PHYSICS_SR)              |    |
| CSNLN                  | [CVPR 20](https://arxiv.org/pdf/2006.01424.pdf)            | [PyTorch](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)         | Cross-Scale Non-Local Attention,Exhaustive Self-Exemplars Mining, Similar to PANet  |
| TTSR                   | [CVPR 20](https://arxiv.org/pdf/2006.04139.pdf)            | [PyTorch](https://github.com/FuzhiYang/TTSR)         | Texture Transformer |
| NSR                    | [arxiv](https://arxiv.org/pdf/2006.04357.pdf)            | [PyTorch](https://github.com/ychfan/nsr)         | Neural Sparse Representation |
| RFANet                 | [CVPR 20](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Residual_Feature_Aggregation_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf)    | [PyTorch](https://github.com/njulj/RFANet)| state-of-the-art SISR |
| Correction filter      | [CVPR 20](http://openaccess.thecvf.com/content_CVPR_2020/papers/Abu_Hussein_Correction_Filter_for_Single_Image_Super-Resolution_Robustifying_Off-the-Shelf_Deep_Super-Resolvers_CVPR_2020_paper.pdf)  |[PyTorch](https://github.com/shadyabh/Correction-Filter)| Enhance  SISR model generalization |
| Unpaired SR            | [CVPR 20](http://openaccess.thecvf.com/content_CVPR_2020/papers/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.pdf)            |    |Unpaired Image Super-Resolution |
| STARnet                | [CVPR 20](http://openaccess.thecvf.com/content_CVPR_2020/papers/Haris_Space-Time-Aware_Multi-Resolution_Video_Enhancement_CVPR_2020_paper.pdf)            |[PyTorch](https://github.com/alterzero/STARnet) | Space-Time-Aware multi-Resolution|
| SSSR                   | [CVPR 20](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.pdf)    | [code](https://github.com/wanglixilinx/DSRL)   |SISR for Semantic Segmentation and Human pose estimation |
| VSR_TGA                | [CVPR 20](http://openaccess.thecvf.com/content_CVPR_2020/papers/Isobe_Video_Super-Resolution_With_Temporal_Group_Attention_CVPR_2020_paper.pdf)    | [code](https://github.com/junpan19/VSR_TGA)   | Temporal Group Attention, Fast Spatial Alignment |
| SSEN                   | [CVPR 20](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.pdf)    |   | Similarity-Aware Deformable Convolution |
| SMSR                   | [arxiv](https://arxiv.org/pdf/2006.09603.pdf)    |[PyTorch](https://github.com/LongguangWang/SMSR)| Sparse Masks, Efficient SISR
| LF-InterNet            | [ECCV 20](https://arxiv.org/pdf/1912.07849.pdf)    | [PyTorch](https://github.com/YingqianWang/LF-InterNet)  | Spatial-Angular Interaction, Light Field Image SR |
| Invertible-Image-Rescaling  | [ECCV 20](https://arxiv.org/abs/2005.05650)    | [Code](https://github.com/pkuxmq/Invertible-Image-Rescaling)  | ECCV oral |
| IGNN                   | [arxiv](https://arxiv.org/abs/2006.16673)    | [Code](https://github.com/sczhou/IGNN)  | GNN, SISR |
| MIRNet                 | [ECCV 20](https://arxiv.org/pdf/2003.06792.pdf)    | [PyTorch](https://github.com/swz30/MIRNet)  | multi-scale residual block |
| SFM                    | [ECCV 20](https://arxiv.org/pdf/2003.07119.pdf)    | [PyTorch](https://github.com/majedelhelou/SFM)  | stochastic frequency mask |
| TCSVT-LightWeight      | [TCSVT](https://arxiv.org/pdf/2007.05835.pdf)    | [TensorFlow](https://github.com/avisekiit/TCSVT-LightWeight-CNNs)  | LightWeight modules |
| PISR                   | [ECCV 20](https://arxiv.org/pdf/2007.07524.pdf)    | [PyTorch](https://github.com/cvlab-yonsei/PISR)  | FSRCNN,distillation framework, HR privileged information |
| MuCAN                  | [ECCV 20](https://arxiv.org/pdf/2007.11803.pdf)    | | **VideoSR**, Temporal Multi-Correspondence Aggregation  |
| DGP                    | [ECCV 20](https://arxiv.org/pdf/2003.13659.pdf)    |[PyTorch](https://github.com/XingangPan/deep-generative-prior) | ECCV oral, GAN, Image Restoration and Manipulation,   |
| RSDN                   | [ECCV 20](https://arxiv.org/pdf/2008.00455.pdf)    |[Code](https://github.com/junpan19/RSDN) | **VideoSR**, Recurrent Neural Network, TwoStream Block|
| CDC                    | [ECCV 20](https://arxiv.org/pdf/2008.01928.pdf)    |[PyTorch](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution) | Diverse Real-world SR dataset, Component Divide-and-Conquer model, GradientWeighted loss|
| MS3-Conv               | [arxiv](https://arxiv.org/pdf/2008.00239.pdf)    | | Multi-Scale cross-Scale Share-weights convolution |
| OverNet                | [arxiv](https://arxiv.org/pdf/2008.02382.pdf)    |  | Lightweight, Overscaling Module, multi-scale loss, Arbitrary Scale Factors |
| RRN                    | [BMVC20](https://arxiv.org/pdf/2008.05765.pdf)    | [code](https://github.com/junpan19/RRN) | **VideoSR**, Recurrent Residual Network, temporal modeling method |
| NAS-DIP                | [ECCV 20](https://arxiv.org/pdf/2008.11713.pdf)    |[PyTorch](https://github.com/YunChunChen/NAS-DIP-pytorch/tree/master/DIP) | NAS|
| SRFlow                 | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500698.pdf)    |[code](https://github.com/andreas128/SRFlow) | Spotlight, Normalizing Flow|
| LatticeNet             | [ECCV 20](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670273.pdf)  | |Lattice Block, LatticeNet, Lightweight, Attention|
| BSRN                   | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490086.pdf)  | |Model Quantization, Binary Neural Network, Bit-Accumulation Mechanism|
| VarSR                  | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680426.pdf)  | |Variational Super-Resolution, very low resolution |
| HAN                    | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570188.pdf)  |[PyTorch](https://github.com/wwlCape/HAN) |SISR, holistic attention network, channel-spatial attention module |
| DeepTemporalSR         | [ECCV 20](http://www.wisdom.weizmann.ac.il/~vision/DeepTemporalSR/supplementary/AcrossScalesAndDimensions_ECCV2020.pdf)  | |Temporal Super-Resolution |
| DGDML-SR               | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620256.pdf)  | |Zero-Shot, Depth Guided Internal Degradation Learning  |
|MLSR                    | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720749.pdf)  |[TensorFlow](https://github.com/parkseobin/MLSR) |Meta-learning, Patch recurrence  |
|PlugNet                 | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600154.pdf)  |[PyTorch](https://github.com/huiyang865/plugnet) |Scene Text Recognition, Feature Squeeze Module |
|TextZoom                | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550647.pdf)  |[code](https://github.com/JasonBoy1/TextZoom) |Scene Text Recognition |
|TPSR                    | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710086.pdf)  |  |NAS,Tiny Perceptual SR |
|CUCaNet                 | [ECCV 20](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740205.pdf)  | [PyTorch](https://github.com/danfenghong/ECCV2020_CUCaNet) |Coupled unmixing, cross-attention,hyperspectral super-resolution, multispectral, unsupervised |
|MAFFSRN                 | [ECCVW 20](https://arxiv.org/pdf/2008.12912.pdf)  |[Code](https://github.com/AbdulMoqeet/MAFFSRN)  |Multi-Attentive Feature Fusion, Ultra Lightweight |
|SRResCycGAN             | [ECCVW 20](https://arxiv.org/pdf/2009.03693.pdf)  | [PyTorch](https://github.com/RaoUmer/SRResCycGAN) |RealSR, CycGAN |
|A-CubeNet               | [arxiv](https://arxiv.org/pdf/2009.05907.pdf)  |  |SISR, lightweight|
|MoG-DUN                 | [arxiv](https://arxiv.org/pdf/2009.06254.pdf)  |  |SISR |
|Understanding Deformable Alignment| [arxiv](https://arxiv.org/pdf/2009.07265.pdf)  | | **VideoSR**, EDVR, offset-fidelity loss |
|AdderSR                 | [arxiv](https://arxiv.org/pdf/2009.08891.pdf)  | |  SISR, adder neural networks, Energy Efficient |
|RFDN                    | [arxiv](https://arxiv.org/pdf/2009.11551.pdf)  | [PyTorch](https://github.com/njulj/RFDN) |  SISR, Lightweight, IMDN, AIM20 WINNER |
|Tarsier                 | [arxiv](https://arxiv.org/pdf/2009.12177.pdf)  | | improve NESRGAN+,injected noise, Diagonal CMA optimize   |
|DeFiAN                  | [arxiv](https://arxiv.org/pdf/2009.13134.pdf)  | [PyTorch](https://github.com/YuanfeiHuang/DeFiAN) |SISR, detail-fidelity attention, Hessian filtering  |
|ASDN                    | [arxiv](https://arxiv.org/pdf/2010.02414.pdf)  | | Arbitrary Scale SR |
|DAN                     | [NeurIPS 20](https://arxiv.org/pdf/2010.02631.pdf)  |[PyTorch](https://github.com/greatlog/DAN)  | Unfolding the Alternating Optimization  |
|DKC                     | [ECCVW 20](https://arxiv.org/pdf/2010.00154.pdf)  | | Deformable Kernel Convolutional, VSR  |
|FAN                     | [ECCVW 20](https://arxiv.org/pdf/2009.14547.pdf)  | | Frequency aggregation network, RealSR  |
|PAN                     | [ECCVW 20](https://arxiv.org/pdf/2010.01073.pdf)  |[PyTorch](https://github.com/zhaohengyuan1/PAN) | Lightweight, Pixel Attention  |
|SCHN                    | [arxiv](https://arxiv.org/pdf/2009.12461.pdf)  | | Blind SR,  Spatial Context Hallucination |
|A2F-SR                  | [ACCV 20](https://arxiv.org/pdf/2011.06773.pdf)  |[PyTorch](https://github.com/wxxxxxxh/A2F-SR) | Lightweight, Attentive Auxiliary Feature Learning  |
|IPT                     | [arxiv](https://arxiv.org/pdf/2012.00364.pdf)  | | Pre-Trained Image Processing Transformer, Imagenet pretrained, dramatically improve performance  |
|GLEAN                   | [arxiv](https://arxiv.org/pdf/2012.00739.pdf)  | | Latent Bank, large scale factor  |
|BasicVSR                | [arxiv](https://arxiv.org/pdf/2012.02181.pdf)  | | **VideoSR**, The Search for Essential Components  |
|EVRNet                  | [arxiv](https://arxiv.org/pdf/2012.02228.pdf)  | | **VideoSR**, design on Edge Devices |
|HRAN                    | [arxiv](https://arxiv.org/pdf/2012.04578.pdf)  | | lightweight,SISR |
|FCA                     | [arxiv](https://arxiv.org/pdf/2012.10102.pdf)  | | Frequency Consistent Adaptation, Real SR |
|DAQ                     | [arxiv](https://arxiv.org/pdf/2012.11230.pdf)  | | Distribution-Aware Quantization |
|HiNAS                   | [arxiv](https://arxiv.org/pdf/2012.13212.pdf)  | | NAS, layerwise architecture sharing strategy, a 1080Ti search, gradient based |

### 2021
More years papers, plase check Quick navigation

| Model                  | Published                                                    | Code                                                         | Keywords                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| TrilevelNAS            | [arxiv](https://arxiv.org/pdf/2101.06658.pdf)            | -              | Trilevel Architecture Search Space      |
| SplitSR                | [arxiv](https://arxiv.org/pdf/2101.07996.pdf)            | -              | lightweight,on Mobile Devices      |
| NODE-SR                | [arxiv](https://arxiv.org/pdf/2101.08987.pdf)            | -              | progressively restore, Neural Differential Equation      |
| BurstSR                | [arxiv](https://arxiv.org/pdf/2101.10997.pdf)            | -              | multi-frame sr, new BurstSR dataset      |
