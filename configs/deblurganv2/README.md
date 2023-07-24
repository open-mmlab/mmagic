# DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better

Code for this paper [DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826)

Orest Kupyn, Tetiana Martyniuk, Junru Wu, Zhangyang Wang

In ICCV 2019

<!-- [ALGORITHM] -->

## Overview

We present a new end-to-end generative adversarial network (GAN) for single image motion deblurring, named 
DeblurGAN-v2, which considerably boosts state-of-the-art deblurring efficiency, quality, and flexibility. DeblurGAN-v2 
is based on a relativistic conditional GAN with a double-scale discriminator. For the first time, we introduce the 
Feature Pyramid Network into deblurring, as a core building block in the generator of DeblurGAN-v2. It can flexibly 
work with a wide range of backbones, to navigate the balance between performance and efficiency. The plug-in of 
sophisticated backbones (e.g., Inception-ResNet-v2) can lead to solid state-of-the-art deblurring. Meanwhile,
with light-weight backbones (e.g., MobileNet and its variants), DeblurGAN-v2 reaches 10-100 times faster than 
the nearest competitors, while maintaining close to state-of-the-art results, implying the option of real-time 
video deblurring. We demonstrate that DeblurGAN-v2 obtains very competitive performance on several popular 
benchmarks, in terms of deblurring quality (both objective and subjective), as well as efficiency. Besides,
we show the architecture to be effective for general image restoration tasks too.

![](https://github.com/VITA-Group/doc_images/kohler_visual.png)
![](https://github.com/VITA-Group/doc_images/restore_visual.png)
![](https://github.com/VITA-Group/doc_images/gopro_table.png)
![](https://github.com/VITA-Group/doc_images/lai_table.png)
<!---![](https://github.com/VITA-Group/doc_images/dvd_table.png)-->
<!---![](https://github.com/VITA-Group/doc_images/kohler_table.png)-->

## DeblurGAN-v2 Architecture

![](./doc_images/pipeline.jpg)

## Predict

```shell
python mmagic/demo/mmagic_inference_demo.py \
        --model-name deblurganv2 \
        --model-comfig ../configs/deblurganv2/deblurganv2_fpn_inception.py \
        --model-ckpt your_ckpt_path \
        --img your_test_image_path \
        --device cpu \
        --result-out-dir ./out.png
```

## Datasets

The datasets for training can be downloaded via the links below:
- [DVD](https://drive.google.com/file/d/1bpj9pCcZR_6-AHb5aNnev5lILQbH8GMZ/view)
- [GoPro](https://drive.google.com/file/d/1KStHiZn5TNm2mo3OLZLjnRvd0vVFCI0W/view)
- [NFS](https://drive.google.com/file/d/1Ut7qbQOrsTZCUJA_mJLptRMipD8sJzjy/view)

## Pre-trained models

<table align="center">
    <tr>
        <th>Dataset</th>
        <th>G Model</th>
        <th>D Model</th>
        <th>Loss Type</th>
        <th>PSNR/ SSIM</th>
        <th>Link</th>
    </tr>
    <tr>
        <td rowspan="3">GoPro Test Dataset</td>
        <td>InceptionResNet-v2</td>
        <td>double_gan</td>
        <td>ragan-ls</td>
        <td>29.55/ 0.934</td>
        <td><a href="https://drive.google.com/uc?export=view&id=1UXcsRVW-6KF23_TNzxw-xC0SzaMfXOaR">fpn_inception.h5</a></td>
    </tr>
    <tr>
        <td>MobileNet</td>
        <td>double_gan</td>
        <td>ragan-ls</td>
        <td>28.17/ 0.925</td>
        <td><a href="https://drive.google.com/uc?export=view&id=1JhnT4BBeKBBSLqTo6UsJ13HeBXevarrU">fpn_mobilenet.h5</a></td>
    </tr>
    <tr>
        <td>MobileNet-DSC</td>
        <td>double_gan</td>
        <td>ragan-ls</td>
        <td>28.03/ 0.922</td>
        <td><a href=""></a></td>
    </tr>
</table>

## Parent Repository

The code was taken from <a href="">https://github.com/KupynOrest/RestoreGAN</a> . This repository contains flexible pipelines for different Image Restoration tasks.

## Citation

If you use this code for your research, please cite our paper.

```
​```
@InProceedings{Kupyn_2019_ICCV,
author = {Orest Kupyn and Tetiana Martyniuk and Junru Wu and Zhangyang Wang},
title = {DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
​```
```
