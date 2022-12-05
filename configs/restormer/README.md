# Restormer (CVPR 2022 -- Oral)

> [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)

> **Task**: Image Restoration

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).

<!-- [IMAGE] -->

<div align=center >
 <img src="https://camo.githubusercontent.com/3c863280a592f3535e4c2411946db0e3495b567fe400bd97f5a8b46f2d3cdcef/68747470733a2f2f692e696d6775722e636f6d2f756c4c6f4569672e706e67" width="800"/>
</div >

## Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

The model weights of different image restoration tasks are downloaded [here(o6un)](https://pan.baidu.com/s/1TWWR_x0gRvfZoK7Fux6prQ?pwd=o6un).

<details>
<summary>Deraining</summary>

| Method                                                                                                                                 | Dataset  | PSNR    | SSIM   | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100.py)   | Test100  | 32.0291 | 0.9239 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Rain100H](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Rain100H.py) | Rain100H | 31.4800 | 0.9056 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Rain100L](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Rain100L.py) | Rain100L | 39.1022 | 0.9787 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test2800](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test2800.py) | Test2800 | 34.2170 | 0.9451 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test1200](/configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test1200.py) | Test1200 | 33.2253 | 0.9272 | 1        | -        |
| -                                                                                                                                      | Average  | 34.0107 | 0.9361 | 1        | -        |

</details>

<details>
<summary>Deblurring</summary>

| Method                                                                                                                                    |            | PSNR    | SSIM   | GPU Info | Download |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro](/configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py)         | GoPro      | 32.9295 | 0.9404 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_HIDE](/configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_HIDE.py)           | HIDE       | 31.2289 | 0.9231 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_RealBlurR](/configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_RealBlurR.py) | RealBlur-R | 35.9141 | 0.9707 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_RealBlurR](/configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_RealBlurJ.py) | RealBlur-J | 28.4356 | 0.8681 | 1        | -        |

</details>

<details>
<summary>Denoising</summary>

**Test Grayscale Gaussian Noise**

| Method                                                                                                                                 | Type                      | Set12        | PSNR    | SSIM   | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma15.py) | Each noise level Training | $\\sigma$=15 | 33.4659 | 0.9137 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma25.py) |                           | $\\sigma$=25 | 31.0880 | 0.8397 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma50.py) |                           | $\\sigma$=50 | 28.0214 | 0.8127 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma15.py) | Blind                     | $\\sigma$=15 | 33.4035 | 0.9125 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma25.py) |                           | $\\sigma$=25 | 31.0880 | 0.8770 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Set12_GNoise_sigma50.py) |                           | $\\sigma$=50 | 28.0145 | 0.8125 | 1        | -        |

| Method                                                                                                                                 | Type                      | BSD68        | PSNR    | SSIM   | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py) | Each noise level Training | $\\sigma$=15 | 31.9535 | 0.8974 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma25.py) |                           | $\\sigma$=25 | 29.5355 | 0.8397 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma50.py) |                           | $\\sigma$=50 | 26.6082 | 0.7422 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py) | Blind                     | $\\sigma$=15 | 30.6039 | 0.8385 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma25.py) |                           | $\\sigma$=25 | 28.0272 | 0.7525 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma50.py) |                           | $\\sigma$=50 | 25.0556 | 0.6174 | 1        | -        |

| Method                                                                                                                                 | Type                      | Urban100     | PSNR    | SSIM   | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma15.py) | Each noise level Training | $\\sigma$=15 | 34.4359 | 0.9420 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma25.py) |                           | $\\sigma$=25 | 32.1171 | 0.9141 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma50.py) |                           | $\\sigma$=50 | 28.9632 | 0.8570 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma15.py) | Blind                     | $\\sigma$=15 | 34.3165 | 0.9410 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma25.py) |                           | $\\sigma$=25 | 32.0490 | 0.9132 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_GNoise_sigma50.py) |                           | $\\sigma$=50 | 28.9847 | 0.8571 | 1        | -        |

**Test Color Gaussian Noise**

| Method                                                                                                                                 | Type                      | CBSD68       | PSNR    | SSIM   | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py) | Each noise level Training | $\\sigma$=15 | 34.3513 | 0.9352 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma25.py) |                           | $\\sigma$=25 | 31.7534 | 0.8945 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma50.py) |                           | $\\sigma$=50 | 28.5511 | 0.8125 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py) | Blind                     | $\\sigma$=15 | 34.3404 | 0.9356 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma25.py) |                           | $\\sigma$=25 | 31.7414 | 0.8946 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma50.py) |                           | $\\sigma$=50 | 28.5506 | 0.8127 | 1        | -        |

| Method                                                                                                                                 | Type                      | Kodak24      | PSNR    | SSIM   | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma15.py) | Each noise level Training | $\\sigma$=15 | 35.4924 | 0.9312 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma25.py) |                           | $\\sigma$=25 | 33.0608 | 0.8943 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma50.py) |                           | $\\sigma$=50 | 30.0220 | 0.8241 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma15.py) | Blind                     | $\\sigma$=15 | 35.4573 | 0.9308 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma25.py) |                           | $\\sigma$=25 | 33.0368 | 0.8939 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak24_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Kodak_CNoise_sigma50.py) |                           | $\\sigma$=50 | 30.0118 | 0.8238 | 1        | -        |

| Method                                                                                                                                 | Type                      | McMaster     | PSNR    | SSIM   | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma15.py) | Each noise level Training | $\\sigma$=15 | 35.6177 | 0.9353 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma25.py) |                           | $\\sigma$=25 | 33.3405 | 0.9069 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma50.py) |                           | $\\sigma$=50 | 30.2738 | 0.8521 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma15.py) | Blind                     | $\\sigma$=15 | 35.5634 | 0.9345 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma25.py) |                           | $\\sigma$=25 | 33.3111 | 0.9064 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_McMaster_CNoise_sigma50.py) |                           | $\\sigma$=50 | 30.2751 | 0.8522 | 1        | -        |

| Method                                                                                                                                 | Type                      | Urban100     | PSNR    | SSIM   | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------ | ------- | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma15.py) | Each noise level Training | $\\sigma$=15 | 35.1535 | 0.9530 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma25.py) |                           | $\\sigma$=25 | 32.9670 | 0.9317 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma50.py) |                           | $\\sigma$=50 | 30.0252 | 0.8902 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma15](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma15.py) | Blind                     | $\\sigma$=15 | 35.0756 | 0.9524 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma25](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma25.py) |                           | $\\sigma$=25 | 32.9174 | 0.9312 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_Color_sigma50](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Urban100_CNoise_sigma50.py) |                           | $\\sigma$=50 | 30.0189 | 0.8897 | 1        | -        |

**Real Image Denoising**

| Method                                                                                                                         | Datasets | PSNR    | SSIM   | MAE    | GPU Info | Download |
| ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------- | ------ | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_SIDD](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SIDD.py) | SIDD     | 40.0156 | 0.9603 | 0.0086 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_DND](/configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SIDD.py)  | DND      | -       | -      | -      | 1        | -        |

</details>

<details>
<summary>Defocus Deblurring</summary>

| Method                                                                                                                                             | Datasets | PSNR    | SSIM   | MAE    | GPU Info | Download |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------- | ------ | ------ | -------- | -------- |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD](/configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py) | DPDD_S   | 25.9805 | 0.8166 | 0.0378 | 1        | -        |
| [restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD](/configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD.py)     | DPDD_D   | 26.6160 | 0.8346 | 0.0354 | 1        | -        |

</details>

## Quick Start

**Train**

**Train Instructions**

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
# Motion Deblurring
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py

# Deraining
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100.py

# Defocus Deblurring
#Single Image
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py
#Dual Image
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD.py

# Denoising
# Color Gaussian Noise
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py
# Grayscale Gaussian Noise
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py

# single-gpu train
# Motion Deblurring
python tools/train.py configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py

# Deraining
python tools/train.py configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100.py

# Defocus Deblurring
#Single Image
python tools/train.py configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py
#Dual Image
python tools/train.py configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD.py

# Denoising
# Color Gaussian Noise
python tools/train.py configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py
# Grayscale Gaussian Noise
python tools/train.py configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py

# multi-gpu train
# Motion Deblurring
./tools/dist_train.sh configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py 8

# Deraining
./tools/dist_train.sh configs/restormer/Deraining/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Test100.py 8

# Defocus Deblurring
#Single Image
./tools/dist_train.sh configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_SingleDPDD.py 8
#Dual Image
./tools/dist_train.sh configs/restormer/DefocusDeblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_DualDPDD.py 8

# Denoising
# Color Gaussian Noise
./tools/dist_train.sh configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_BSD68_GNoise_sigma15.py 8
# Grayscale Gaussian Noise
./tools/dist_train.sh configs/restormer/Denoising/restormer_d48nb4668nrb4h1248-lr3e-4-300k_CBSD68_CNoise_sigma15.py 8

```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

**Test**

Test Instructions

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
# Motion Deblurring
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py Motion_Deblurring_checkpoint_path

# single-gpu test
# Motion Deblurring
python tools/test.py configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py Motion_Deblurring_checkpoint_path



# multi-gpu test
# Motion Deblurring
./tools/dist_test.sh configs/restormer/Deblurring/restormer_d48nb4668nrb4h1248-lr3e-4-300k_Gopro.py Motion_Deblurring_checkpoint_path

```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

## Citation

```bibtex
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration},
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```
