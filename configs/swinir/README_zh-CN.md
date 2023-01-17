# SwinIR (ICCVW'2021)

> **任务**: 图像超分辨率, 图像去噪, JPEG压缩伪影移除

<!-- [ALGORITHM] -->

<details>
<summary align="right">SwinIR (ICCVW'2021)</summary>

```bibtex
@inproceedings{liang2021swinir,
  title={Swinir: Image restoration using swin transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1833--1844},
  year={2021}
}
```

</details>

<br/>

### **Classical Image Super-Resolution**

在 Y 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                算法                                 | Set5 PSNR | Set14 PSNR | DIV2K PSNR | Set5 SSIM | Set14 SSIM | DIV2K SSIM | GPU 信息 |                                下载                                 |
| :-----------------------------------------------------------------: | :-------: | :--------: | :--------: | :-------: | :--------: | :--------: | :------: | :-----------------------------------------------------------------: |
| [swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k](./swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  38.3240  |  34.1174   |  37.8921   |  0.9626   |   0.9230   |   0.9481   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k-ed2d419e.pth) \| log |
| [swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k](./swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  34.8640  |  30.7669   |  34.1397   |  0.9317   |   0.8508   |   0.8917   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k-926950f1.pth) \| log |
| [swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k](./swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  32.7315  |  28.9065   |  32.0953   |  0.9029   |   0.7915   |   0.8418   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k-88e4903d.pth) \| log |
| [swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k](./swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  38.3971  |  34.4149   |  37.9473   |  0.9629   |   0.9252   |   0.9488   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k-69e15fb6.pth) \| log |
| [swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k](./swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  34.9335  |  30.9258   |  34.2830   |  0.9323   |   0.8540   |   0.8939   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k-d6982f7b.pth) \| log |
| [swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k](./swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  32.9214  |  29.0792   |  32.3021   |  0.9053   |   0.7953   |   0.8451   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k-0502d775.pth) \| log |

### **Lightweight Image Super-Resolution**

在 Y 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                算法                                 | Set5 PSNR | Set14 PSNR | DIV2K PSNR | Set5 SSIM | Set14 SSIM | DIV2K SSIM | GPU 信息 |                                下载                                 |
| :-----------------------------------------------------------------: | :-------: | :--------: | :--------: | :-------: | :--------: | :--------: | :------: | :-----------------------------------------------------------------: |
| [swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k](./swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  38.1289  |  33.8404   |  37.5844   |  0.9617   |   0.9207   |   0.9459   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k-131d3f64.pth) \| log |
| [swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k](./swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  34.6037  |  30.5340   |  33.8394   |  0.9293   |   0.8468   |   0.8867   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k-309cb239.pth) \| log |
| [swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k](./swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  32.4343  |  28.7441   |  31.8636   |  0.8984   |   0.7861   |   0.8353   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k-d6622d03.pth) \| log |

### **Real-World Image Super-Resolution**

在 Y 通道上进行评估。
我们使用 NIQE 作为指标。

|                                         算法                                          | RealSRSet+5images NIQE | GPU 信息 |                                         下载                                          |
| :-----------------------------------------------------------------------------------: | :--------------------: | :------: | :-----------------------------------------------------------------------------------: |
| [swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](./swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) |         5.7975         |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_gan-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-c6425057.pth) \| log |
| [swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](./swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) |         7.2738         |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x2s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-6f0c425f.pth) \| log |
| [swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](./swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) |         5.2329         |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-36960d18.pth) \| log |
| [swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost](./swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-ost.py) |         7.7460         |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d6e180_8xb4-lr1e-4-600k_df2k-os-a016a72f.pth) \| log |
| [swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost](./swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py) |         5.1464         |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-9f1599b5.pth) \| log |
| [swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost](./swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py) |         7.6378         |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_psnr-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-25f1722a.pth) \| log |

### **Grayscale Image Deoising**

在灰度图上进行评估。
我们使用 PSNR 作为指标。

|                                      算法                                       | Set12 PSNR | BSD68 PSNR | Urban100 PSNR | GPU 信息 |                                       下载                                       |
| :-----------------------------------------------------------------------------: | :--------: | :--------: | :-----------: | :------: | :------------------------------------------------------------------------------: |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15](./swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15.py) |  33.9731   |  32.5203   |    34.3424    |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN15-6782691b.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25](./swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25.py) |  31.6434   |  30.1377   |    31.9493    |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN25-d0d8d4da.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50](./swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50.py) |  28.5651   |  27.3157   |    28.6626    |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-grayDN50-54c9968a.pth) \| log |

### **Color Image Deoising**

在 RGB 通道上进行评估。
我们使用 PSNR 作为指标。

|                                  算法                                   | CBSD68 PSNR | Kodak24 PSNR | McMaster PSNR | Urban100 PSNR | GPU 信息 |                                   下载                                   |
| :---------------------------------------------------------------------: | :---------: | :----------: | :-----------: | :-----------: | :------: | :----------------------------------------------------------------------: |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15](./swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15.py) |   34.4136   |   35.3555    |    35.6205    |    35.1836    |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN15-c74a2cee.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25](./swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25.py) |   31.7626   |   32.9003    |    33.3198    |    32.9458    |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN25-df2b1c0c.pth) \| log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50](./swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50.py) |   28.5346   |   29.8058    |    30.2027    |    29.8832    |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_dfwb-colorDN50-e369874c.pth) \| log |

### **JPEG Compression Artifact Reduction (grayscale)**

在灰度图上进行评估。
我们使用 PSNR 和 SSIM 作为指标。

|                                   算法                                    | Classic5 PSNR | Classic5 SSIM | LIVE1 PSNR | LIVE1 SSIM | GPU 信息 |                                   下载                                    |
| :-----------------------------------------------------------------------: | :-----------: | :-----------: | :--------: | :--------: | :------: | :-----------------------------------------------------------------------: |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10](./swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10.py) |    30.2746    |    0.8254     |  29.8611   |   0.8292   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR10-da93c8e9.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20](./swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20.py) |    32.5331    |    0.8753     |  32.2667   |   0.8914   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR20-d47367b1.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30](./swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30.py) |    33.7504    |    0.8966     |  33.7001   |   0.9179   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR30-52c083cf.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40](./swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40.py) |    34.5377    |    0.9087     |  34.6846   |   0.9322   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-grayCAR40-803e8d9b.pth) \| log |

### **JPEG Compression Artifact Reduction (color)**

在 RGB 通道上进行评估。
我们使用 PSNR 和 SSIM 作为指标。

|                                   算法                                    | Classic5 PSNR | Classic5 SSIM | LIVE1 PSNR | LIVE1 SSIM | GPU 信息 |                                   下载                                    |
| :-----------------------------------------------------------------------: | :-----------: | :-----------: | :--------: | :--------: | :------: | :-----------------------------------------------------------------------: |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10](./swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10.py) |    30.1019    |    0.8217     |  28.0676   |   0.8094   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR10-09aafadc.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20](./swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20.py) |    32.3489    |    0.8727     |  30.3489   |   0.8745   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR20-b8a42b5e.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30](./swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30.py) |    33.6028    |    0.8949     |  31.8235   |   0.9023   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30-e9fe6859.pth) \| log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40](./swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py) |    34.4344    |    0.9076     |  32.7610   |   0.9179   |    8     | [model](https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40-5b77a6e6.pth) \| log |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
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



# 单个GPU上训练
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



# 多个GPU上训练
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

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
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



# 单个GPU上测试
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



# 多GPU测试
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

./tools/dist_test.sh configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR40.py https://download.openmmlab.com/mmediting/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_dfwb-colorCAR30-e9fe6859.pth

```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
