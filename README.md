# NatSR

# [Natural and Realistic Single Image Super-Resolution with Explicit Natural Manifold Discrimination (CVPR 2019)](http://openaccess.thecvf.com/content_CVPR_2019/html/Soh_Natural_and_Realistic_Single_Image_Super-Resolution_With_Explicit_Natural_Manifold_CVPR_2019_paper.html)
<br><br>

Jae Woong Soh, Gu Yong Park, Junho Jo, and Nam Ik Cho

[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Soh_Natural_and_Realistic_Single_Image_Super-Resolution_With_Explicit_Natural_Manifold_CVPR_2019_paper.pdf) [Supplementary](http://openaccess.thecvf.com/content_CVPR_2019/supplemental/Soh_Natural_and_Realistic_CVPR_2019_supplemental.pdf)

## Environments
- Ubuntu 16.04
- [Tensorflow 1.8](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.1
- Python 3.6

## Abstract

Recently, many convolutional neural networks for single image super-resolution (SISR) have been proposed, which focus on reconstructing the high-resolution images in terms of objective distortion measures. However, the networks trained with objective loss functions generally fail to reconstruct the realistic fine textures and details that are essential for better perceptual quality. Recovering the realistic details remains a challenging problem, and only a few works have been proposed which aim at increasing the perceptual quality by generating enhanced textures. However, the generated fake details often make undesirable artifacts and the overall image looks somewhat unnatural. Therefore, in this paper, we present a new approach to reconstructing realistic super-resolved images with high perceptual quality, while maintaining the naturalness of the result. In particular, we focus on the domain prior properties of SISR problem. Specifically, we define the naturalness prior in the low-level domain and constrain the output image in the natural manifold, which eventually generates more natural and realistic images. Our results show better naturalness compared to the recent super-resolution algorithms including perception-oriented ones.
<br><br>

## Related Work

### Distortion Oriented Single Image Super-Resolution

#### [EDSR (CVPRW 2017)] Enhanced Deep Residual Networks for Single Image Super-Resolution <a href="http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/html/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.html">Link</a> 

### Perception Oriented Single Image Super-Resolution

#### [SRGAN (CVPR 2017)] Photo-realistic Single Image Super-Resolution Using a Generative Adversarial Network<a href="http://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html">Link</a> 

#### [EnhanceNet (ICCV 2017)] Enhancenet: Single Image Super-Resolution Through Automated Texture Synthesis <a href="http://openaccess.thecvf.com/content_iccv_2017/html/Sajjadi_EnhanceNet_Single_Image_ICCV_2017_paper.html">Link</a>

#### [SFT-GAN (CVPR 2018)] Recovering Realistic Texture in Image Super-Resolution by Deep Spatial Feature Transform <a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Recovering_Realistic_Texture_CVPR_2018_paper.html">Link</a>
<br><br>

## Brief Description of Our Proposed Method

### Explicitly Modeling the SISR & Designing Natural Manifold

<p align="center"><img src="figure/Manifold.png" width="900"></p>

As inifinite number of high-resolution (HR) images can be correspond to one low-resolution (LR) image, SISR is one-to-many problem.
Therefore, we first define HR space and we divided HR space into three subspaces based on our prior knowledge.

### Natural Manifold Discrimination

<p align="center"><img src="figure/NMD.png" width="900"></p>

The network architecture of Natural Manifold Discriminator (NMD) and the loss function for training NMD.

### Natural and Realistic Single Image Super-Resolution (NatSR)

<p align="center"><img src="figure/Overall.png" width="400">&nbsp;&nbsp;<img src="figure/SRNet.png" width="400"></p> 

Left: The overall training scheme & Right: The SR network (Generator) architecture.

## Experimental Results

**(FR-IQA) Results of the average PSNR (dB) and SSIM for the benchmark**

<p align="center"><img src="figure/FR-IQA.png" width="900"></p>

**(NR-IQA) Results of the [NIQE](https://ieeexplore.ieee.org/abstract/document/6353522/) and [NQSR](https://www.sciencedirect.com/science/article/pii/S107731421630203X) for BSD100 (Left)**
**Perception-Distortion Tradeoff [Ref](http://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html) Plot for BSD100 (Right)**

<p align="center"><img src="figure/NR-IQA.png" width="400">&nbsp;&nbsp;<img src="figure/Tradeoff.png" width="400"></p> 


## Visualized Results

<p align="center"><img src="figure/1.png" width="400">&nbsp;&nbsp;<img src="figure/2.png" width="400"></p>
<br><br>
<p align="center"><img src="figure/3.png" width="900"></p>

## Guideline for Test Codes

**Requisites should be installed beforehand.**

Clone this repo.

```
git clone http://github.com/JWSoh/NatSR.git
cd NatSR/
```

Ready for input data (low-resolution).

**Test**

[Options]
```
python test.py --gpu [GPU_number] --ref [True/False] --datapath [LR path] --labelpath [HR path] --modelpath [pretrained model path] --model [NatSR/FRSR] --savepath [SR path] --save [True/False]
<br><br>
--gpu: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
--ref: [True/False] True if there exist reference images. (Reference images are just for PSNR measurements.) [Default True]
--datapath: Path of input images.
--labelpath: Path of reference images. (Not required, only for PSNR.)
--modelpath: Path of pretrained models. (If you clone this repo., you don't need to specify.
--model: [NatSR/FRSR] The type of model. [Default: NatSR]
--savepath: Path for super-resolved images. [Default: result]
--save: [True/False] Flag whether to save SR images. [Default: True]

```

Examples

To generate super-resolved image and also to measure PSNR.
```
python test.py --gpu 0 --datapath LR/Set5 --labelpath HR/Set5 --model NatSR
```
To generate output images, only.

```
python test.py --gpu 0 --ref 0 --datapath LR/Set5 --model NatSR
```

Super-resolve with FRSR (Distortion oriented model)

```
python test.py --gpu 0 --ref 0 --datapath LR/Set5 --model FRSR
```


## Citation
```
@InProceedings{Soh_2019_CVPR,
author = {Soh, Jae Woong and Park, Gu Yong and Jo, Junho and Cho, Nam Ik},
title = {Natural and Realistic Single Image Super-Resolution With Explicit Natural Manifold Discrimination},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```