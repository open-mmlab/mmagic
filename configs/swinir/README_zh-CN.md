# SwinIR (ICCVW'2021)

> **任务**: 图像恢复

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

001 Image Super-Resolution
在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                     算法                                      | Set5 PSNR | Set14 PSNR | Set5 SSIM | Set14 SSIM | GPU 信息 |                                     下载                                      |
| :---------------------------------------------------------------------------: | :-------: | :--------: | :-------: | :--------: | :------: | :---------------------------------------------------------------------------: |
| [swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  36.2104  |  32.3254   |  0.9458   |   0.8997   |    1     | [model](https://drive.google.com/file/d/1Uw1QtPKnBvFalgx8sC-moFmTLCBtfbqM/view?usp=share_link) \\ log |
| [swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  32.9243  |  29.0268   |  0.9080   |   0.8212   |    1     | [model](https://drive.google.com/file/d/13VOC_15bH4OcfqLX3TYa3NwoTSADKH9b/view?usp=share_link) \\ log |
| [swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k](/configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py) |  30.9975  |  27.2786   |  0.8757   |   0.8973   |    1     | [model](https://drive.google.com/file/d/12IXYTR_3UebYbTqR9wumBEv5AlL6Qyr9/view?usp=share_link) \\ log |
| [swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  36.1395  |  32.0471   |  0.9453   |   0.9398   |    1     | [model](https://drive.google.com/file/d/1NwpDavsYKNcVptQyUrCzFp2ArAtBV6a2/view?usp=share_link) \\ log |
| [swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  32.8619  |  28.8743   |  0.9073   |   0.8178   |    1     | [model](https://drive.google.com/file/d/11fn_CkgaYl-flzaeJeapKa0d17RwPNQ9/view?usp=share_link) \\ log |
| [swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py) |  30.8093  |  27.1080   |  0.8729   |   0.7540   |    1     | [model](https://drive.google.com/file/d/1KWaJ3X6ZrXJZ37jHczdjRTcElQ_sPTpP/view?usp=share_link) \\ log |
| [swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  35.9517  |  31.7807   |  0.9442   |   0.8948   |    1     | [model](https://drive.google.com/file/d/13dDwSMxjBpZZiXlgKHH9onkxzLUKZ0LX/view?usp=share_link) \\ log |
| [swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  32.6025  |  28.6482   |  0.9045   |   0.8136   |    1     | [model](https://drive.google.com/file/d/1Jj0Mdyd2sbaaredwNxVtp0zraHr_EgCN/view?usp=share_link) \\ log |
| [swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k](/configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py) |  30.5225  |  26.9440   |  0.8678   |   0.7484   |    1     | [model](https://drive.google.com/file/d/1hf-Bod4nAo13dRgyHKYiAi260a1sYCT8/view?usp=share_link) \\ log |

003 Real-World Image Super-Resolution

|                                               算法                                               | GPU 信息 |                                               下载                                               |
| :----------------------------------------------------------------------------------------------: | :------: | :----------------------------------------------------------------------------------------------: |
| [swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost](/configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py) |    1     | [model](https://drive.google.com/file/d/1efvIxFkevJpRsUd-Mvq7OIeeGbBhXtyJ/view?usp=share_link) \\ log |
| [swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost](/configs/swinir/swinir_x2s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py) |    1     | [model](https://drive.google.com/file/d/1RpeouyxHZbhS0z-uoX8r5Ys2jD8R59IL/view?usp=share_link) \\ log |
| [swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost](/configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-gan_df2k-ost.py) |    1     | [model](https://drive.google.com/file/d/1YryTalO1rqlqtVi7ZSnEGXXqg_Pg4_9w/view?usp=share_link) \\ log |
| [swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost](/configs/swinir/swinir_x4s64w8d6e180_8xb4-lr1e-4-600k-psnr_df2k-ost.py) |    1     | [model](https://drive.google.com/file/d/1T0XYcNdAVM_UuQlgauHrUSLM_EPB_XCY/view?usp=share_link) \\ log |
| [swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-gan_df2k-ost](/configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-gan_df2k-ost.py) |    1     | [model](https://drive.google.com/file/d/1RJOir_JbWcjZDKK1Rq6GqDtrE4DQhbKL/view?usp=share_link) \\ log |
| [swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-psnr_df2k-ost](/configs/swinir/swinir_x4s64w8d9e240_8xb4-lr1e-4-600k-psnr_df2k-ost.py) |    1     | [model](https://drive.google.com/file/d/1YAeG9duFEt5yJtFS_iCFI685mgb7UUJi/view?usp=share_link) \\ log |

004 Grayscale Image Deoising

|                                      算法                                       | Set12 PSNR | BSD68 PSNR | Urban100 PSNR | GPU 信息 |                                       下载                                       |
| :-----------------------------------------------------------------------------: | :--------: | :--------: | :-----------: | :------: | :------------------------------------------------------------------------------: |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN15_dfwb](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN15_dfwb.py) |  33.9731   |  32.5203   |    34.3424    |    1     | [model](https://drive.google.com/file/d/18PmDIFYZtlvyLQnGYvKShO-CdtlPwIwe/view?usp=share_link) \\ log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN25_dfwb](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN25_dfwb.py) |  31.6434   |  30.1377   |    31.9493    |    1     | [model](https://drive.google.com/file/d/1PqC9a-3wfyH6DeVpP2yi6r6stqEB1vfj/view?usp=share_link) \\ log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN50_dfwb](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_grayDN50_dfwb.py) |  28.5651   |  27.3157   |    28.6626    |    1     | [model](https://drive.google.com/file/d/1miDCBmxe73XoxkJDLAmqOBbuiY9U6JRz/view?usp=share_link) \\ log |

005 Color Image Deoising

|                                  算法                                   | CBSD68 PSNR | Kodak24 PSNR | McMaster PSNR | Urban100 PSNR | GPU 信息 |                                   下载                                   |
| :---------------------------------------------------------------------: | :---------: | :----------: | :-----------: | :-----------: | :------: | :----------------------------------------------------------------------: |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN15_dfwb.py) |   34.4136   |   35.3555    |    35.6205    |    35.1836    |    1     | [model](https://drive.google.com/file/d/16pfIBzAXTv6-3xTsKXEtLkY2pLM-tNFn/view?usp=share_link) \\ log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN25_dfwb.py) |   31.7626   |   32.9003    |    33.3198    |    32.9458    |    1     | [model](https://drive.google.com/file/d/1pihZhiw1V5hWNoaWCuabN8KCj6_il6KG/view?usp=share_link) \\ log |
| [swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN50_dfwb](/configs/swinir/swinir_s128w8d6e180_8xb1-lr2e-4-1600k_colorDN50_dfwb.py) |   28.5346   |   29.8058    |    30.2027    |    29.8832    |    1     | [model](https://drive.google.com/file/d/1pihZhiw1V5hWNoaWCuabN8KCj6_il6KG/view?usp=share_link) \\ log |

006 JPEG Compression Artifact Reduction (grayscale)

|                                        算法                                         | classic5 PSNR | classic5 SSIM | GPU 信息 |                                        下载                                         |
| :---------------------------------------------------------------------------------: | :-----------: | :-----------: | :------: | :---------------------------------------------------------------------------------: |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR10_dfwb](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR10_dfwb.py) |    30.2746    |    0.8254     |    1     | [model](https://drive.google.com/file/d/1LMEGlGtYcrJ9dhC8pmNsCAWyeZq1Dfc5/view?usp=share_link) \\ log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR20_dfwb](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR20_dfwb.py) |    32.5331    |    0.8753     |    1     | [model](https://drive.google.com/file/d/1624dceqoBD5CwqLuL_ozPaYuIQex2WmL/view?usp=share_link) \\ log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR30_dfwb.py) |    33.7504    |    0.8966     |    1     | [model](https://drive.google.com/file/d/1X70GSCK8Wo9nYUmtN0MysCarIHliB3WM/view?usp=share_link) \\ log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR40_dfwb](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_grayCAR40_dfwb.py) |    34.5377    |    0.9087     |    1     | [model](https://drive.google.com/file/d/1HQGYXthHnmVsng1313KZmGzewsWPzzDT/view?usp=share_link) \\ log |

006 JPEG Compression Artifact Reduction (color)

|                                          算法                                          | LIVE1 PSNR | LIVE1 SSIM | GPU 信息 |                                          下载                                          |
| :------------------------------------------------------------------------------------: | :--------: | :--------: | :------: | :------------------------------------------------------------------------------------: |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR10_dfwb](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR10_dfwb.py) |  27.9085   |   0.8057   |    1     | [model](https://drive.google.com/file/d/1YXlmXo5SdQF7JwEvedUmNgxrPZVwxAgg/view?usp=share_link) \\ log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR20_dfwb](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR20_dfwb.py) |  30.2844   |   0.8717   |    1     | [model](https://drive.google.com/file/d/1pY6RXvp2y98Qx4VhaZOSVTpyOQzk8l9b/view?usp=share_link) \\ log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR30_dfwb](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR30_dfwb.py) |  31.6396   |   0.9000   |    1     | [model](https://drive.google.com/file/d/1LUiFJCdcGCRl4NDAtiyj2Ba87fKW3p5x/view?usp=share_link) \\ log |
| [swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR40_dfwb](/configs/swinir/swinir_s126w7d6e180_8xb1-lr2e-4-1600k_colorCAR40_dfwb.py) |  32.5833   |   0.9158   |    1     | [model](https://drive.google.com/file/d/1N7veqPM2ypF1hOyq8UC7BHLb4K5391wZ/view?usp=share_link) \\ log |

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



# 单个GPU上测试
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



# 多个GPU上测试
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

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
