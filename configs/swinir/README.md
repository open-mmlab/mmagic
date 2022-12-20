# SwinIR (ICCVW'2021)

> [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

> **Task**: Image Super-Resolution,

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Image restoration is a long-standing low-level vision problem that aims to restore high-quality images from low-quality images (e.g., downscaled, noisy and compressed images). While state-of-the-art image restoration methods are based on convolutional neural networks, few attempts have been made with Transformers which show impressive performance on high-level vision tasks. In this paper, we propose a strong baseline model SwinIR for image restoration based on the Swin Transformer. SwinIR consists of three parts: shallow feature extraction, deep feature extraction and high-quality image reconstruction. In particular, the deep feature extraction module is composed of several residual Swin Transformer blocks (RSTB), each of which has several Swin Transformer layers together with a residual connection. We conduct experiments on three representative tasks: image super-resolution (including classical, lightweight and real-world image super-resolution), image denoising (including grayscale and color image denoising) and JPEG compression artifact reduction. Experimental results demonstrate that SwinIR outperforms state-of-the-art methods on different tasks by up to 0.14~0.45dB, while the total number of parameters can be reduced by up to 67%.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/40970489/204525707-272fb8c6-1bb3-41f2-9a9b-612c48ddd9b4.png" width="800"/>
</div >

## Results and models

### **Classical Image Super-Resolution**

Evaluated on Y channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                               Method                               | Set5 PSNR | Set14 PSNR | DIV2K PSNR | Set5 SSIM | Set14 SSIM | DIV2K SSIM | GPU Info |                               Download                               |
| :----------------------------------------------------------------: | :-------: | :--------: | :--------: | :-------: | :--------: | :--------: | :------: | :------------------------------------------------------------------: |
| [swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  38.3240  |  34.1174   |  37.8921   |  0.9626   |   0.9230   |   0.9481   |    8     | [model](https://drive.google.com/file/d/1NwpDavsYKNcVptQyUrCzFp2ArAtBV6a2/view?usp=share_link) \| log |
| [swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  34.8640  |  30.7669   |  34.1397   |  0.9317   |   0.8508   |   0.8917   |    8     | [model](https://drive.google.com/file/d/11fn_CkgaYl-flzaeJeapKa0d17RwPNQ9/view?usp=share_link) \| log |
| [swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  32.7315  |  28.9065   |  32.0953   |  0.9029   |   0.7915   |   0.8418   |    8     | [model](https://drive.google.com/file/d/1KWaJ3X6ZrXJZ37jHczdjRTcElQ_sPTpP/view?usp=share_link) \| log |
| [swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  38.3971  |  34.4149   |  37.9473   |  0.9629   |   0.9252   |   0.9488   |    8     | [model](https://drive.google.com/file/d/1Uw1QtPKnBvFalgx8sC-moFmTLCBtfbqM/view?usp=share_link) \| log |
| [swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  34.9335  |  30.9258   |  34.2830   |  0.9323   |   0.8540   |   0.8939   |    8     | [model](https://drive.google.com/file/d/13VOC_15bH4OcfqLX3TYa3NwoTSADKH9b/view?usp=share_link) \| log |
| [swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  32.9214  |  29.0792   |  32.3021   |  0.9053   |   0.7953   |   0.8451   |    8     | [model](https://drive.google.com/file/d/12IXYTR_3UebYbTqR9wumBEv5AlL6Qyr9/view?usp=share_link) \| log |

### **Lightweight Image Super-Resolution**

Evaluated on Y channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                               Method                               | Set5 PSNR | Set14 PSNR | DIV2K PSNR | Set5 SSIM | Set14 SSIM | DIV2K SSIM | GPU Info |                               Download                               |
| :----------------------------------------------------------------: | :-------: | :--------: | :--------: | :-------: | :--------: | :--------: | :------: | :------------------------------------------------------------------: |
| [swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  38.1289  |  33.8404   |  37.5844   |  0.9617   |   0.9207   |   0.9459   |    8     | [model](https://drive.google.com/file/d/13dDwSMxjBpZZiXlgKHH9onkxzLUKZ0LX/view?usp=share_link) \| log |
| [swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  34.6037  |  30.5340   |  33.8394   |  0.9293   |   0.8468   |   0.8867   |    8     | [model](https://drive.google.com/file/d/1Jj0Mdyd2sbaaredwNxVtp0zraHr_EgCN/view?usp=share_link) \| log |
| [swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  32.4343  |  28.7441   |  31.8636   |  0.8984   |   0.7861   |   0.8353   |    8     | [model](https://drive.google.com/file/d/1hf-Bod4nAo13dRgyHKYiAi260a1sYCT8/view?usp=share_link) \| log |

### **Real-World Image Super-Resolution**

Evaluated on Y channels.
The metrics are `NIQE` .

|                                        Method                                        | RealSRSet+5images NIQE | GPU Info |                                        Download                                        |
| :----------------------------------------------------------------------------------: | :--------------------: | :------: | :------------------------------------------------------------------------------------: |
| [swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) |         5.7975         |    8     | [model](https://drive.google.com/file/d/1efvIxFkevJpRsUd-Mvq7OIeeGbBhXtyJ/view?usp=share_link) \| log |
| [swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) |         7.2738         |    8     | [model](https://drive.google.com/file/d/1RpeouyxHZbhS0z-uoX8r5Ys2jD8R59IL/view?usp=share_link) \| log |
| [swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) |         5.2329         |    8     | [model](https://drive.google.com/file/d/1YryTalO1rqlqtVi7ZSnEGXXqg_Pg4_9w/view?usp=share_link) \| log |
| [swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) |         7.7460         |    8     | [model](https://drive.google.com/file/d/1T0XYcNdAVM_UuQlgauHrUSLM_EPB_XCY/view?usp=share_link) \| log |
| [swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py) |         5.1464         |    8     | [model](https://drive.google.com/file/d/1RJOir_JbWcjZDKK1Rq6GqDtrE4DQhbKL/view?usp=share_link) \| log |
| [swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py) |         7.6378         |    8     | [model](https://drive.google.com/file/d/1YAeG9duFEt5yJtFS_iCFI685mgb7UUJi/view?usp=share_link) \| log |

### **Grayscale Image Deoising**

Evaluated on grayscale images.
The metrics are `PSNR` .

|                                     Method                                     | Set12 PSNR | BSD68 PSNR | Urban100 PSNR | GPU Info |                                     Download                                      |
| :----------------------------------------------------------------------------: | :--------: | :--------: | :-----------: | :------: | :-------------------------------------------------------------------------------: |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py) |  33.9731   |  32.5203   |    34.3424    |    8     | [model](https://drive.google.com/file/d/18PmDIFYZtlvyLQnGYvKShO-CdtlPwIwe/view?usp=share_link) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py) |  31.6434   |  30.1377   |    31.9493    |    8     | [model](https://drive.google.com/file/d/1PqC9a-3wfyH6DeVpP2yi6r6stqEB1vfj/view?usp=share_link) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py) |  28.5651   |  27.3157   |    28.6626    |    8     | [model](https://drive.google.com/file/d/1miDCBmxe73XoxkJDLAmqOBbuiY9U6JRz/view?usp=share_link) \| log |

### **Color Image Deoising**

Evaluated on RGB channels.
The metrics are `PSNR` .

|                                 Method                                 | CBSD68 PSNR | Kodak24 PSNR | McMaster PSNR | Urban100 PSNR | GPU Info |                                 Download                                  |
| :--------------------------------------------------------------------: | :---------: | :----------: | :-----------: | :-----------: | :------: | :-----------------------------------------------------------------------: |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py) |   34.4136   |   35.3555    |    35.6205    |    35.1836    |    8     | [model](https://drive.google.com/file/d/16pfIBzAXTv6-3xTsKXEtLkY2pLM-tNFn/view?usp=share_link) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py) |   31.7626   |   32.9003    |    33.3198    |    32.9458    |    8     | [model](https://drive.google.com/file/d/1pihZhiw1V5hWNoaWCuabN8KCj6_il6KG/view?usp=share_link) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py) |   28.5346   |   29.8058    |    30.2027    |    29.8832    |    8     | [model](https://drive.google.com/file/d/1pihZhiw1V5hWNoaWCuabN8KCj6_il6KG/view?usp=share_link) \| log |

### **JPEG Compression Artifact Reduction (grayscale)**

Evaluated on grayscale images.
The metrics are `PSNR` .

|                                  Method                                  | Classic5 PSNR | Classic5 SSIM | LIVE1 PSNR | LIVE1 SSIM | GPU Info |                                  Download                                  |
| :----------------------------------------------------------------------: | :-----------: | :-----------: | :--------: | :--------: | :------: | :------------------------------------------------------------------------: |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py) |    30.2746    |    0.8254     |  29.8611   |   0.8292   |    8     | [model](https://drive.google.com/file/d/1LMEGlGtYcrJ9dhC8pmNsCAWyeZq1Dfc5/view?usp=share_link) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py) |    32.5331    |    0.8753     |  32.2667   |   0.8914   |    8     | [model](https://drive.google.com/file/d/1624dceqoBD5CwqLuL_ozPaYuIQex2WmL/view?usp=share_link) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py) |    33.7504    |    0.8966     |  33.7001   |   0.9179   |    8     | [model](https://drive.google.com/file/d/1X70GSCK8Wo9nYUmtN0MysCarIHliB3WM/view?usp=share_link) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py) |    34.5377    |    0.9087     |  34.6846   |   0.9322   |    8     | [model](https://drive.google.com/file/d/1HQGYXthHnmVsng1313KZmGzewsWPzzDT/view?usp=share_link) \| log |

### **JPEG Compression Artifact Reduction (color)**

Evaluated on RGB channels.
The metrics are `PSNR` .

|                                  Method                                  | Classic5 PSNR | Classic5 SSIM | LIVE1 PSNR | LIVE1 SSIM | GPU Info |                                  Download                                  |
| :----------------------------------------------------------------------: | :-----------: | :-----------: | :--------: | :--------: | :------: | :------------------------------------------------------------------------: |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py) |    30.1019    |    0.8217     |  27.9085   |   0.8057   |    8     | [model](https://drive.google.com/file/d/1YXlmXo5SdQF7JwEvedUmNgxrPZVwxAgg/view?usp=share_link) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py) |    32.3489    |    0.8727     |  30.2844   |   0.8717   |    8     | [model](https://drive.google.com/file/d/1pY6RXvp2y98Qx4VhaZOSVTpyOQzk8l9b/view?usp=share_link) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py) |    33.6028    |    0.8949     |  31.6396   |   0.9000   |    8     | [model](https://drive.google.com/file/d/1LUiFJCdcGCRl4NDAtiyj2Ba87fKW3p5x/view?usp=share_link) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py) |    34.4344    |    0.9076     |  32.5833   |   0.9158   |    8     | [model](https://drive.google.com/file/d/1N7veqPM2ypF1hOyq8UC7BHLb4K5391wZ/view?usp=share_link) \| log |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py

# 002 Lightweight Image Super-Resolution (small size)
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py

# 003 Real-World Image Super-Resolution
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-gan_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-psnr_df2k-ost.py

# 004 Grayscale Image Deoising (middle size)
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN15_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN25_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN50_dfwb.py

# 005 Color Image Deoising (middle size)
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN50_dfwb.py

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR10_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR20_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR40_dfwb.py

# color
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR10_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR20_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR30_dfwb.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR40_dfwb.py



# single-gpu train
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
python tools/train.py configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py
python tools/train.py configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py
python tools/train.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
python tools/train.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py
python tools/train.py configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py
python tools/train.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py

# 002 Lightweight Image Super-Resolution (small size)
python tools/train.py configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py
python tools/train.py configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py
python tools/train.py configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py

# 003 Real-World Image Super-Resolution
python tools/train.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py
python tools/train.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py
python tools/train.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py
python tools/train.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py
python tools/train.py configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-gan_df2k-ost.py
python tools/train.py configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-psnr_df2k-ost.py

# 004 Grayscale Image Deoising (middle size)
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN15_dfwb.py
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN25_dfwb.py
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN50_dfwb.py

# 005 Color Image Deoising (middle size)
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb.py
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb.py
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN50_dfwb.py

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR10_dfwb.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR20_dfwb.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR40_dfwb.py

# color
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR10_dfwb.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR20_dfwb.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR30_dfwb.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR40_dfwb.py



# multi-gpu train
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
./tools/dist_train.sh configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py 8
./tools/dist_train.sh configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py 8
./tools/dist_train.sh configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py 8

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
./tools/dist_train.sh configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py 8
./tools/dist_train.sh configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py 8
./tools/dist_train.sh configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py 8

# 002 Lightweight Image Super-Resolution (small size)
./tools/dist_train.sh configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py 8
./tools/dist_train.sh configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py 8
./tools/dist_train.sh configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py 8

# 003 Real-World Image Super-Resolution
./tools/dist_train.sh configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-gan_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-psnr_df2k-ost.py 8

# 004 Grayscale Image Deoising (middle size)
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN15_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN25_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN50_dfwb.py 8

# 005 Color Image Deoising (middle size)
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN50_dfwb.py 8

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR10_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR20_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR40_dfwb.py 8

# color
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR10_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR20_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR30_dfwb.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR40_dfwb.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth

# 002 Lightweight Image Super-Resolution (small size)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth

# 003 Real-World Image Super-Resolution
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth

# 004 Grayscale Image Deoising (middle size)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN15_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN25_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN50_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth

# 005 Color Image Deoising (middle size)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN50_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR10_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR20_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR40_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth

# color
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR10_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR20_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR30_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR40_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth



# single-gpu test
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
python tools/test.py configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth
python tools/test.py configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth
python tools/test.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
python tools/test.py configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth
python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth

# 002 Lightweight Image Super-Resolution (small size)
python tools/test.py configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth
python tools/test.py configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth
python tools/test.py configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth

# 003 Real-World Image Super-Resolution
python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth
python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR.pth
python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth
python tools/test.py configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
python tools/test.py configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth

# 004 Grayscale Image Deoising (middle size)
python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN15_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth
python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN25_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth
python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN50_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth

# 005 Color Image Deoising (middle size)
python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth
python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth
python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN50_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR10_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR20_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR40_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth

# color
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR10_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR20_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR30_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR40_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth



# multi-gpu test
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
./tools/dist_test.sh configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth 8
./tools/dist_test.sh configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth 8
./tools/dist_test.sh configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth 8

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
./tools/dist_test.sh configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth 8
./tools/dist_test.sh configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth 8
./tools/dist_test.sh configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py /path/to/checkpoint/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth 8

# 002 Lightweight Image Super-Resolution (small size)
./tools/dist_test.sh configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth 8
./tools/dist_test.sh configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth 8
./tools/dist_test.sh configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py /path/to/checkpoint/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth 8

# 003 Real-World Image Super-Resolution
./tools/dist_test.sh configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth 8
./tools/dist_test.sh configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR.pth 8
./tools/dist_test.sh configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth 8
./tools/dist_test.sh configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth 8
./tools/dist_test.sh configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-gan_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth 8
./tools/dist_test.sh configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-psnr_df2k-ost.py /path/to/checkpoint/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_PSNR.pth 8

# 004 Grayscale Image Deoising (middle size)
./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN15_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth 8
./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN25_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth 8
./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN50_dfwb.py /path/to/checkpoint/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth 8

# 005 Color Image Deoising (middle size)
./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth 8
./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth 8
./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN50_dfwb.py /path/to/checkpoint/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth 8

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR10_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth 8
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR20_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth 8
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth 8
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR40_dfwb.py /path/to/checkpoint/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth 8

# color
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR10_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth 8
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR20_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth 8
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR30_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth 8
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR40_dfwb.py /path/to/checkpoint/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>

## Citation

```bibtex
@inproceedings{liang2021swinir,
  title={Swinir: Image restoration using swin transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1833--1844},
  year={2021}
}
```
