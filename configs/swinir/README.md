# SwinIR (ICCVW'2021)

> [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

> **Task**: Image Super-Resolution, Image denoising, JPEG compression artifact reduction

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

|                               Model                                | Dataset |          Task          | Scale |  PSNR   |  SSIM  | Training Resources |                               Download                                |
| :----------------------------------------------------------------: | :-----: | :--------------------: | :---: | :-----: | :----: | :----------------: | :-------------------------------------------------------------------: |
| [swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  Set5   | Image Super-Resolution |  x2   | 38.3240 | 0.9626 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k-ed2d419e.pth) \| log |
| [swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  Set14  | Image Super-Resolution |  x2   | 34.1174 | 0.9230 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k-ed2d419e.pth) \| log |
| [swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  DIV2K  | Image Super-Resolution |  x2   | 37.8921 | 0.9481 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k-ed2d419e.pth) \| log |
| [swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  Set5   | Image Super-Resolution |  x3   | 34.8640 | 0.9317 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k-926950f1.pth) \| log |
| [swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  Set14  | Image Super-Resolution |  x3   | 30.7669 | 0.8508 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k-926950f1.pth) \| log |
| [swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  DIV2K  | Image Super-Resolution |  x3   | 34.1397 | 0.8917 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k-926950f1.pth) \| log |
| [swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  Set5   | Image Super-Resolution |  x4   | 32.7315 | 0.9029 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k-88e4903d.pth) \| log |
| [swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  Set14  | Image Super-Resolution |  x4   | 28.9065 | 0.7915 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k-88e4903d.pth) \| log |
| [swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  DIV2K  | Image Super-Resolution |  x4   | 32.0953 | 0.8418 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k-88e4903d.pth) \| log |
| [swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  Set5   | Image Super-Resolution |  x2   | 38.3971 | 0.9629 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k-69e15fb6.pth) \| log |
| [swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  Set14  | Image Super-Resolution |  x2   | 34.4149 | 0.9252 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k-69e15fb6.pth) \| log |
| [swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  DIV2K  | Image Super-Resolution |  x2   | 37.9473 | 0.9488 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k-69e15fb6.pth) \| log |
| [swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  Set5   | Image Super-Resolution |  x3   | 34.9335 | 0.9323 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k-d6982f7b.pth) \| log |
| [swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  Set14  | Image Super-Resolution |  x3   | 30.9258 | 0.8540 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k-d6982f7b.pth) \| log |
| [swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  DIV2K  | Image Super-Resolution |  x3   | 34.2830 | 0.8939 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k-d6982f7b.pth) \| log |
| [swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  Set5   | Image Super-Resolution |  x4   | 32.9214 | 0.9053 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k-0502d775.pth) \| log |
| [swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  Set14  | Image Super-Resolution |  x4   | 29.0792 | 0.7953 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k-0502d775.pth) \| log |
| [swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  DIV2K  | Image Super-Resolution |  x4   | 32.3021 | 0.8451 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k-0502d775.pth) \| log |

### **Lightweight Image Super-Resolution**

Evaluated on Y channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                               Model                                | Dataset |          Task          | Scale |  PSNR   |  SSIM  | Training Resources |                               Download                                |
| :----------------------------------------------------------------: | :-----: | :--------------------: | :---: | :-----: | :----: | :----------------: | :-------------------------------------------------------------------: |
| [swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  Set5   | Image Super-Resolution |  x2   | 38.1289 | 0.9617 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k-131d3f64.pth) \| log |
| [swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  Set14  | Image Super-Resolution |  x2   | 33.8404 | 0.9207 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k-131d3f64.pth) \| log |
| [swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  DIV2K  | Image Super-Resolution |  x2   | 37.5844 | 0.9459 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k-131d3f64.pth) \| log |
| [swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  Set5   | Image Super-Resolution |  x3   | 34.6037 | 0.9293 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k-309cb239.pth) \| log |
| [swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  Set14  | Image Super-Resolution |  x3   | 30.5340 | 0.8468 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k-309cb239.pth) \| log |
| [swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  DIV2K  | Image Super-Resolution |  x3   | 33.8394 | 0.8867 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k-309cb239.pth) \| log |
| [swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  Set5   | Image Super-Resolution |  x4   | 32.4343 | 0.8984 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth) \| log |
| [swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  Set14  | Image Super-Resolution |  x4   | 28.7441 | 0.7861 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth) \| log |
| [swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  DIV2K  | Image Super-Resolution |  x4   | 31.8636 | 0.8353 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth) \| log |

### **Real-World Image Super-Resolution**

Evaluated on Y channels.
The metrics are `NIQE` .

|                                Model                                |      Dataset      |          Task          |  NIQE  | Training Resources |                                Download                                |
| :-----------------------------------------------------------------: | :---------------: | :--------------------: | :----: | :----------------: | :--------------------------------------------------------------------: |
| [swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) | RealSRSet+5images | Image Super-Resolution | 5.7975 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-c6425057.pth) \| log |
| [swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) | RealSRSet+5images | Image Super-Resolution | 7.2738 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-6f0c425f.pth) \| log |
| [swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) | RealSRSet+5images | Image Super-Resolution | 5.2329 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-36960d18.pth) \| log |
| [swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) | RealSRSet+5images | Image Super-Resolution | 7.7460 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-a016a72f.pth) \| log |
| [swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py) | RealSRSet+5images | Image Super-Resolution | 5.1464 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-9f1599b5.pth) \| log |
| [swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost](/configs/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py) | RealSRSet+5images | Image Super-Resolution | 7.6378 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-9f1599b5.pth) \| log |

### **Grayscale Image Deoising**

Evaluated on grayscale images.
The metrics are `PSNR` .

|                                   Model                                    | Dataset  |      Task       |  PSNR   | Training Resources |                                    Download                                    |
| :------------------------------------------------------------------------: | :------: | :-------------: | :-----: | :----------------: | :----------------------------------------------------------------------------: |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py) |  Set12   | Image denoising | 33.9731 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15-6782691b.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py) |  BSD68   | Image denoising | 32.5203 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15-6782691b.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py) | Urban100 | Image denoising | 34.3424 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15-6782691b.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py) |  Set12   | Image denoising | 31.6434 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25-d0d8d4da.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py) |  BSD68   | Image denoising | 30.1377 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25-d0d8d4da.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py) | Urban100 | Image denoising | 31.9493 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25-d0d8d4da.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py) |  Set12   | Image denoising | 28.5651 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50-54c9968a.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py) |  BSD68   | Image denoising | 27.3157 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50-54c9968a.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py) | Urban100 | Image denoising | 28.6626 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50-54c9968a.pth) \| log |

### **Color Image Deoising**

Evaluated on RGB channels.
The metrics are `PSNR` .

|                                   Model                                    | Dataset  |      Task       |  PSNR   | Training Resources |                                    Download                                    |
| :------------------------------------------------------------------------: | :------: | :-------------: | :-----: | :----------------: | :----------------------------------------------------------------------------: |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py) |  CBSD68  | Image denoising | 34.4136 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py) | Kodak24  | Image denoising | 35.3555 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py) | McMaster | Image denoising | 35.6205 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py) | Urban100 | Image denoising | 35.1836 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py) |  CBSD68  | Image denoising | 31.7626 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py) | Kodak24  | Image denoising | 32.9003 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py) | McMaster | Image denoising | 33.3198 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py) | Urban100 | Image denoising | 32.9458 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py) |  CBSD68  | Image denoising | 28.5346 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py) | Kodak24  | Image denoising | 29.8058 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py) | McMaster | Image denoising | 30.2027 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py) | Urban100 | Image denoising | 29.8832 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth) \| log |

### **JPEG Compression Artifact Reduction (grayscale)**

Evaluated on grayscale images.
The metrics are \`PSNR / SSIM

|                             Model                             | Dataset  |                Task                 |  PSNR   |  SSIM  | Training Resources |                             Download                              |
| :-----------------------------------------------------------: | :------: | :---------------------------------: | :-----: | :----: | :----------------: | :---------------------------------------------------------------: |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py) | Classic5 | JPEG compression artifact reduction | 30.2746 | 0.8254 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10-da93c8e9.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py) |  LIVE1   | JPEG compression artifact reduction | 29.8611 | 0.8292 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10-da93c8e9.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py) | Classic5 | JPEG compression artifact reduction | 32.5331 | 0.8753 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20-d47367b1.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py) |  LIVE1   | JPEG compression artifact reduction | 32.2667 | 0.8914 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20-d47367b1.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py) | Classic5 | JPEG compression artifact reduction | 33.7504 | 0.8966 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30-52c083cf.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py) |  LIVE1   | JPEG compression artifact reduction | 33.7001 | 0.9179 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30-52c083cf.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py) | Classic5 | JPEG compression artifact reduction | 34.5377 | 0.9087 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40-803e8d9b.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py) |  LIVE1   | JPEG compression artifact reduction | 34.6846 | 0.9322 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40-803e8d9b.pth) \| log |

### **JPEG Compression Artifact Reduction (color)**

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                             Model                             | Dataset  |                Task                 |  PSNR   |  SSIM  | Training Resources |                             Download                              |
| :-----------------------------------------------------------: | :------: | :---------------------------------: | :-----: | :----: | :----------------: | :---------------------------------------------------------------: |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py) | Classic5 | JPEG compression artifact reduction | 30.1019 | 0.8217 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10-09aafadc.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py) |  LIVE1   | JPEG compression artifact reduction | 28.0676 | 0.8094 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10-09aafadc.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py) | Classic5 | JPEG compression artifact reduction | 32.3489 | 0.8727 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20-b8a42b5e.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py) |  LIVE1   | JPEG compression artifact reduction | 30.4514 | 0.8745 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20-b8a42b5e.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py) | Classic5 | JPEG compression artifact reduction | 33.6028 | 0.8949 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30-e9fe6859.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py) |  LIVE1   | JPEG compression artifact reduction | 31.8235 | 0.9023 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30-e9fe6859.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py) | Classic5 | JPEG compression artifact reduction | 34.4344 | 0.9076 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40-5b77a6e6.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py) |  LIVE1   | JPEG compression artifact reduction | 32.7610 | 0.9179 |         8          | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40-5b77a6e6.pth) \| log |

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
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py

# 004 Grayscale Image Deoising (middle size)
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py

# 005 Color Image Deoising (middle size)
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py

# color
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py


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
python tools/train.py configs/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py
python tools/train.py configs/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py
python tools/train.py configs/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py
python tools/train.py configs/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py
python tools/train.py configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py
python tools/train.py configs/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py

# 004 Grayscale Image Deoising (middle size)
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py

# 005 Color Image Deoising (middle size)
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py
python tools/train.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py

# color
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py
python tools/train.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py


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
./tools/dist_train.sh configs/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py 8
./tools/dist_train.sh configs/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py 8

# 004 Grayscale Image Deoising (middle size)
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py 8
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py 8
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py 8

# 005 Color Image Deoising (middle size)
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py 8
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py 8
./tools/dist_train.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py 8

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py 8

# color
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py 8
./tools/dist_train.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k-ed2d419e.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k-926950f1.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k-88e4903d.pth

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k-69e15fb6.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k-d6982f7b.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k-0502d775.pth


# 002 Lightweight Image Super-Resolution (small size)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k-131d3f64.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k-309cb239.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth

# 003 Real-World Image Super-Resolution
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-c6425057.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-6f0c425f.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-36960d18.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-a016a72f.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-9f1599b5.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-25f1722a.pth

# 004 Grayscale Image Deoising (middle size)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15-6782691b.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25-d0d8d4da.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50-54c9968a.pth

# 005 Color Image Deoising (middle size)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding usesx8 blocks)
# grayscale
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10-da93c8e9.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20-d47367b1.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30-52c083cf.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40-803e8d9b.pth


# color
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10-09aafadc.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20-b8a42b5e.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30-e9fe6859.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40-5b77a6e6.pth



# single-gpu test
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
python tools/test.py configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k-ed2d419e.pth

python tools/test.py configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k-926950f1.pth

python tools/test.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k-88e4903d.pth

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k-69e15fb6.pth

python tools/test.py configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k-d6982f7b.pth

python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k-0502d775.pth


# 002 Lightweight Image Super-Resolution (small size)
python tools/test.py configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k-131d3f64.pth

python tools/test.py configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k-309cb239.pth

python tools/test.py configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth


# 003 Real-World Image Super-Resolution
python tools/test.py configs/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-c6425057.pth

python tools/test.py configs/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-6f0c425f.pth

python tools/test.py configs/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-36960d18.pth

python tools/test.py configs/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-a016a72f.pth

python tools/test.py configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-9f1599b5.pth

python tools/test.py configs/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-25f1722a.pth


# 004 Grayscale Image Deoising (middle size)
python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15-6782691b.pth

python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25-d0d8d4da.pth

python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50-54c9968a.pth


# 005 Color Image Deoising (middle size)
python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth

python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth

python tools/test.py configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth


# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding usesx8 blocks)
# grayscale
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10-da93c8e9.pth

python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20-d47367b1.pth

python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30-52c083cf.pth

python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40-803e8d9b.pth


# color
python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10-09aafadc.pth

python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20-b8a42b5e.pth

python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30-e9fe6859.pth

python tools/test.py configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40-5b77a6e6.pth



# multi-gpu test
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
./tools/dist_test.sh configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k-ed2d419e.pth

./tools/dist_test.sh configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k-926950f1.pth

./tools/dist_test.sh configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k-88e4903d.pth

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
./tools/dist_test.sh configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k-69e15fb6.pth

./tools/dist_test.sh configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k-d6982f7b.pth

./tools/dist_test.sh configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k-0502d775.pth

# 002 Lightweight Image Super-Resolution (small size)
./tools/dist_test.sh configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k-131d3f64.pth

./tools/dist_test.sh configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k-309cb239.pth

./tools/dist_test.sh configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth

# 003 Real-World Image Super-Resolution
./tools/dist_test.sh configs/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-c6425057.pth

./tools/dist_test.sh configs/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-6f0c425f.pth

./tools/dist_test.sh configs/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-36960d18.pth

./tools/dist_test.sh configs/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-a016a72f.pth

./tools/dist_test.sh configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-9f1599b5.pth

./tools/dist_test.sh configs/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-25f1722a.pth

# 004 Grayscale Image Deoising (middle size)
./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15-6782691b.pth

./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25-d0d8d4da.pth

./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50-54c9968a.pth

# 005 Color Image Deoising (middle size)
./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth

./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth

./tools/dist_test.sh configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth

# 006 JPEG Compression Artifact Reduction (middle size, using window_size=7 because JPEG encoding uses 8x8 blocks)
# grayscale
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10-da93c8e9.pth

./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20-d47367b1.pth

./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30-52c083cf.pth

./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40-803e8d9b.pth

# color
./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10-09aafadc.pth

./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20-b8a42b5e.pth

./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30-e9fe6859.pth

./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py https://download.openmmlab.com/mmediting/swinir/

```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

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
