# Restormer (CVPR'2022)

> [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)

> **Task**: Denoising, Deblurring, Deraining

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Since convolutional neural networks (CNNs) perform well at learning generalizable image priors from large-scale data, these models have been extensively applied to image restoration and related tasks. Recently, another class of neural architectures, Transformers, have shown significant performance gains on natural language and high-level vision tasks. While the Transformer model mitigates the shortcomings of CNNs (i.e., limited receptive field and inadaptability to input content), its computational complexity grows quadratically with the spatial resolution, therefore making it infeasible to apply to most image restoration tasks involving high-resolution images. In this work, we propose an efficient Transformer model by making several key designs in the building blocks (multi-head attention and feed-forward network) such that it can capture long-range pixel interactions, while still remaining applicable to large images. Our model, named Restoration Transformer (Restormer), achieves state-of-the-art results on several image restoration tasks, including image deraining, single-image motion deblurring, defocus deblurring (single-image and dual-pixel data), and image denoising (Gaussian grayscale/color denoising, and real image denoising).

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/49083766/206964466-95de972a-ff92-4493-9097-73118590d78f.png" width="800"/>
</div >

## Results and models

### **Deraining**

Evaluated on Y channels. The metrics are `PSNR` / `SSIM` .

|                             Model                             | Dataset  |   Task    | PSNR (Y) | SSIM (Y) | Training Resources |                                         Download                                         |
| :-----------------------------------------------------------: | :------: | :-------: | :------: | :------: | :----------------: | :--------------------------------------------------------------------------------------: |
| [restormer_official_rain13k](./restormer_official_rain13k.py) | Rain100H | Deraining | 31.4804  |  0.9056  |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth) \| log |
| [restormer_official_rain13k](./restormer_official_rain13k.py) | Rain100L | Deraining | 39.1023  |  0.9787  |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth) \| log |
| [restormer_official_rain13k](./restormer_official_rain13k.py) | Test100  | Deraining | 32.0287  |  0.9239  |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth) \| log |
| [restormer_official_rain13k](./restormer_official_rain13k.py) | Test1200 | Deraining | 33.2251  | /0.9272  |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth) \| log |
| [restormer_official_rain13k](./restormer_official_rain13k.py) | Test2800 | Deraining | 34.2170  |  0.9451  |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth) \| log |

### **Motion Deblurring**

Evaluated on RGB channels for GoPro and HIDE, and Y channel for ReakBlur-J and ReakBlur-R. The metrics are `PSNR` / `SSIM` .

|                           Model                           |  Dataset   |    Task    | PSNR/SSIM (RGB) | <br>PSNR/SSIM (Y) | Training Resources |                                 Download                                  |
| :-------------------------------------------------------: | :--------: | :--------: | :-------------: | :---------------: | :----------------: | :-----------------------------------------------------------------------: |
| [restormer_official_gopro](./restormer_official_gopro.py) |   GoPro    | Deblurring | 32.9295/0.9496  |         -         |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth) \| log |
| [restormer_official_gopro](./restormer_official_gopro.py) |    HIDE    | Deblurring | 31.2289/0.9345  |         -         |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth) \| log |
| [restormer_official_gopro](./restormer_official_gopro.py) | RealBlur-J | Deblurring |        -        |  28.4356/0.8681   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth) \| log |
| [restormer_official_gopro](./restormer_official_gopro.py) | RealBlur-R | Deblurring |        -        |  35.9141/0.9707   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth) \| log |

### **Defocus Deblurring**

Evaluated on RGB channels. The metrics are `PSNR` / `SSIM` / `MAE` / `LPIPS`.

|                                Model                                 |    Dataset     |    Task    |  PSNR   |  SSIM  |  MAE   | Training Resources |                                Download                                 |
| :------------------------------------------------------------------: | :------------: | :--------: | :-----: | :----: | :----: | :----------------: | :---------------------------------------------------------------------: |
| [restormer_official_dpdd-single](./restormer_official_dpdd-single.py) | Indoor Scenes  | Deblurring | 28.8681 | 0.8859 | 0.0251 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth) \| log |
| [restormer_official_dpdd-single](./restormer_official_dpdd-single.py) | Outdoor Scenes | Deblurring | 23.2410 | 0.7509 | 0.0499 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth) \| log |
| [restormer_official_dpdd-single](./restormer_official_dpdd-single.py) |    Combined    | Deblurring | 25.9805 | 0.8166 | 0.0378 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth) \| log |
|  [restormer_official_dpdd-dual](./restormer_official_dpdd-dual.py)   | Indoor Scenes  | Deblurring | 26.6160 | 0.8346 | 0.0354 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth) \| log |
|  [restormer_official_dpdd-dual](./restormer_official_dpdd-dual.py)   | Outdoor Scenes | Deblurring | 26.6160 | 0.8346 | 0.0354 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth) \| log |
|  [restormer_official_dpdd-dual](./restormer_official_dpdd-dual.py)   |    Combined    | Deblurring | 26.6160 | 0.8346 | 0.0354 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth) \| log |

### **Gaussian Denoising**

**Test Grayscale Gaussian Noise**

Evaluated on grayscale images. The metrics are `PSNR` / `SSIM` .

**training a separate model for each noise level**

|                                   Model                                    | Dataset  | Task | $\\sigma$ |  PSNR   |  SSIM  | Training Resources |                                    Download                                    |
| :------------------------------------------------------------------------: | :------:|:-:| | :-------: | :-----: | :----: | :----------------: | :----------------------------------------------------------------------------: |
| [restormer_official_dfwb-gray-sigma15](./restormer_official_dfwb-gray-sigma15.py) |  Set12   |Denoising|    15     | 34.0182 | 0.9160 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth) | log |
| [restormer_official_dfwb-gray-sigma15](./restormer_official_dfwb-gray-sigma15.py) |  BSD68   |Denoising|    15     | 32.4987 | 0.8940 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth) | log |
| [restormer_official_dfwb-gray-sigma15](./restormer_official_dfwb-gray-sigma15.py) | Urban100 |Denoising|    15     | 34.4336 | 0.9419 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth) | log |
| [restormer_official_dfwb-gray-sigma25](./restormer_official_dfwb-gray-sigma25.py) |  Set12   |Denoising|    25     | 31.7289 | 0.8811 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth) | log |
| [restormer_official_dfwb-gray-sigma25](./restormer_official_dfwb-gray-sigma25.py) |  BSD68   |Denoising|    25     | 30.1613 | 0.8370 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth) | log |
| [restormer_official_dfwb-gray-sigma25](./restormer_official_dfwb-gray-sigma25.py) | Urban100 |Denoising|    25     | 32.1162 | 0.9140 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth) | log |
| [restormer_official_dfwb-gray-sigma50](./restormer_official_dfwb-gray-sigma50.py) |  Set12   |Denoising|    50     | 28.6269 | 0.8188 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth) | log |
| [restormer_official_dfwb-gray-sigma50](./restormer_official_dfwb-gray-sigma50.py) |  BSD68   |Denoising|    50     | 27.3266 | 0.7434 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth) | log |
| [restormer_official_dfwb-gray-sigma50](./restormer_official_dfwb-gray-sigma50.py) | Urban100 |Denoising|    50     | 28.9636 | 0.8571 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth) | log |

**learning a single model to handle various noise levels**

|                                 Model                                  | Dataset  |   Task    | $\\sigma$ |  PSNR   |  SSIM  | Training Resources |                                 Download                                  |
| :--------------------------------------------------------------------: | :------: | :-------: | :-------: | :-----: | :----: | :----------------: | :-----------------------------------------------------------------------: |
| [restormer_official_dfwb-gray-sigma15](./restormer_official_dfwb-gray-sigma15.py) |  Set12   | Denoising |    15     | 33.9642 | 0.9153 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma15](./restormer_official_dfwb-gray-sigma15.py) |  BSD68   | Denoising |    15     | 32.4994 | 0.8928 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma15](./restormer_official_dfwb-gray-sigma15.py) | Urban100 | Denoising |    15     | 34.3152 | 0.9409 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma25](./restormer_official_dfwb-gray-sigma25.py) |  Set12   | Denoising |    25     | 31.7106 | 0.8810 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma25](./restormer_official_dfwb-gray-sigma25.py) |  BSD68   | Denoising |    25     | 30.1486 | 0.8360 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma25](./restormer_official_dfwb-gray-sigma25.py) | Urban100 | Denoising |    25     | 32.0457 | 0.9131 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma50](./restormer_official_dfwb-gray-sigma50.py) |  Set12   | Denoising |    50     | 28.6614 | 0.8197 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma50](./restormer_official_dfwb-gray-sigma50.py) |  BSD68   | Denoising |    50     | 27.3537 | 0.7422 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |
| [restormer_official_dfwb-gray-sigma50](./restormer_official_dfwb-gray-sigma50.py) | Urban100 | Denoising |    50     | 28.9848 | 0.8571 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth) \| log |

**Test Color Gaussian Noise**

Evaluated on RGB channels. The metrics are `PSNR` / `SSIM` .
**training a separate model for each noise level**

|                               Model                                | Dataset  |   Task    | $\\sigma$ | PSNR (RGB) | SSIM (RGB) | Training Resources |                                Download                                |
| :----------------------------------------------------------------: | :------: | :-------: | :-------: | :--------: | :--------: | :----------------: | :--------------------------------------------------------------------: |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py) |  CBSD68  | Denoising |    15     |  34.3506   |   0.9352   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth) \| log |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py) | Kodak24  | Denoising |    15     |  35.4900   |   0.9312   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth) \| log |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py) | McMaster | Denoising |    15     |  35.6072   |   0.9352   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth) \| log |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py) | Urban100 | Denoising |    15     |  35.1522   |   0.9530   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth) \| log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py) |  CBSD68  | Denoising |    25     |  31.7457   |   0.8942   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth) \| log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py) | Kodak24  | Denoising |    25     |  33.0489   |   0.8943   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth) \| log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py) | McMaster | Denoising |    25     |  33.3260   |   0.9066   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth) \| log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py) | Urban100 | Denoising |    25     |  32.9670   |   0.9317   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth) \| log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py) |  CBSD68  | Denoising |    50     |  28.5569   |   0.8127   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth) \| log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py) | Kodak24  | Denoising |    50     |  30.0122   |   0.8238   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth) \| log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py) | McMaster | Denoising |    50     |  30.2608   |   0.8515   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth) \| log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py) | Urban100 | Denoising |    50     |  30.0230   |   0.8902   |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth) \| log |

**learning a single model to handle various noise levels**

|                                  Model                                  | Dataset |Task| $\\sigma$ | PSNR (RGB) | SSIM (RGB) |                                  Training Resources                                   | Download |
| :---------------------------------------------------------------------: | :-----: | :-------: | :--------: | :--------: | :-----------------------------------------------------------------------------------: | :------: |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py)|  CBSD68   |Denoising|   15    |  34.3422  |   0.9356   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py)| Kodak24   |Denoising|   15    |  35.4544  |   0.9308   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py)| McMaster  |Denoising|   15    |  35.5473  |   0.9344   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma15](./restormer_official_dfwb-color-sigma15.py)| Urban100  |Denoising|   15    |  35.0754  |   0.9524   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py)|  CBSD68   |Denoising|   25    |  31.7391  |   0.8945   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py)| Kodak24   |Denoising|   25    |  33.0380  |   0.8941   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py)| McMaster  |Denoising|   25    |  33.3040  |   0.9063   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma25](./restormer_official_dfwb-color-sigma25.py)| Urban100  |Denoising|   25    |  32.9165  |   0.9312   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py)|  CBSD68   |Denoising|   50    |  28.5582  |   0.8126   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py)| Kodak24   |Denoising|   50    |  30.0074  |   0.8233   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py)| McMaster  |Denoising|   50    |  30.2671  |   0.8520   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |
| [restormer_official_dfwb-color-sigma50](./restormer_official_dfwb-color-sigma50.py)| Urban100  |Denoising|   50    |  30.0172  |   0.8898   |     1      | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth) | log |

### **Real Image Denoising**

Evaluated on RGB channels. The metrics are `PSNR` / `SSIM` .

|                          Model                          | Dataset |   Task    |  PSNR   |  SSIM  | Training Resources |                                              Download                                              |
| :-----------------------------------------------------: | :-----: | :-------: | :-----: | :----: | :----------------: | :------------------------------------------------------------------------------------------------: |
| [restormer_official_sidd](./restormer_official_sidd.py) |  SIDD   | Denoising | 40.0156 | 0.9225 |         1          | [model](https://download.openmmlab.com/mmediting/restormer/restormer_official_sidd-9e7025db.pth) \| log |

## Quick Start

**Train**

You can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
# Deraining
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_rain13k.py https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth

# Motion Deblurring
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_gopro.py https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth

# Defocus Deblurring
# Single
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dpdd-dual.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth
# Dual
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dpdd-single.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth

# Gaussian Denoising
# Test Grayscale Gaussian Noise
# sigma15
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma25
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma50
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# Test Color Gaussian Noise
# sigma15
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma25
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma50
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth

CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# single-gpu test
# Deraining
python tools/test.py configs/restormer/restormer_official_rain13k.py https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth

# Motion Deblurring
python tools/test.py configs/restormer/restormer_official_gopro.py https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth

# Defocus Deblurring
# Single
python tools/test.py configs/restormer/restormer_official_dpdd-dual.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth
# Dual
python tools/test.py configs/restormer/restormer_official_dpdd-single.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth

# Gaussian Denoising
# Test Grayscale Gaussian Noise
# sigma15
python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth

python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma25
python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth

python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma50
python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth

python tools/test.py configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# Test Color Gaussian Noise
# sigma15
python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth

python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma25
python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth

python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma50
python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth

python tools/test.py configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth


# multi-gpu test
# Deraining
./tools/dist_test.sh configs/restormer/restormer_official_rain13k.py https://download.openmmlab.com/mmediting/restormer/restormer_official_rain13k-2be7b550.pth

# Motion Deblurring
./tools/dist_test.sh configs/restormer/restormer_official_gopro.py https://download.openmmlab.com/mmediting/restormer/restormer_official_gopro-db7363a0.pth

# Defocus Deblurring
# Single
./tools/dist_test.sh configs/restormer/restormer_official_dpdd-dual.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-single-6bc31582.pth
# Dual
./tools/dist_test.sh configs/restormer/restormer_official_dpdd-single.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dpdd-dual-52c94c00.pth

# Gaussian Denoising
# Test Grayscale Gaussian Noise
# sigma15
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma15-da74417f.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma25
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma25-08010841.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# sigma50
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-sigma50-ee852dfe.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-gray-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-gray-blind-5f094bcc.pth

# Test Color Gaussian Noise
# sigma15
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma15-012ceb71.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma15.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma25
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma25-e307f222.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma25.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

# sigma50
./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-sigma50-a991983d.pth

./tools/dist_test.sh configs/restormer/restormer_official_dfwb-color-sigma50.py https://download.openmmlab.com/mmediting/restormer/restormer_official_dfwb-color-blind-dfd03c9f.pth

```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

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
