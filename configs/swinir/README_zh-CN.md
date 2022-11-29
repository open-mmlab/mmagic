# SwinIR (ICCVW'2021)

> **任务**: 图像超分辨率

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
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1NwpDavsYKNcVptQyUrCzFp2ArAtBV6a2/view?usp=share_link
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/11fn_CkgaYl-flzaeJeapKa0d17RwPNQ9/view?usp=share_link
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1KWaJ3X6ZrXJZ37jHczdjRTcElQ_sPTpP/view?usp=share_link

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/1Uw1QtPKnBvFalgx8sC-moFmTLCBtfbqM/view?usp=share_link
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/13VOC_15bH4OcfqLX3TYa3NwoTSADKH9b/view?usp=share_link
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/12IXYTR_3UebYbTqR9wumBEv5AlL6Qyr9/view?usp=share_link

# 002 Lightweight Image Super-Resolution (small size)
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/13dDwSMxjBpZZiXlgKHH9onkxzLUKZ0LX/view?usp=share_link
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1Jj0Mdyd2sbaaredwNxVtp0zraHr_EgCN/view?usp=share_link
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1hf-Bod4nAo13dRgyHKYiAi260a1sYCT8/view?usp=share_link


# 单个GPU上测试
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
python tools/test.py configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1NwpDavsYKNcVptQyUrCzFp2ArAtBV6a2/view?usp=share_link
python tools/test.py configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/11fn_CkgaYl-flzaeJeapKa0d17RwPNQ9/view?usp=share_link
python tools/test.py configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1KWaJ3X6ZrXJZ37jHczdjRTcElQ_sPTpP/view?usp=share_link

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
python tools/test.py configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/1Uw1QtPKnBvFalgx8sC-moFmTLCBtfbqM/view?usp=share_link
python tools/test.py configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/13VOC_15bH4OcfqLX3TYa3NwoTSADKH9b/view?usp=share_link
python tools/test.py configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/12IXYTR_3UebYbTqR9wumBEv5AlL6Qyr9/view?usp=share_link

# 002 Lightweight Image Super-Resolution (small size)
python tools/test.py configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/13dDwSMxjBpZZiXlgKHH9onkxzLUKZ0LX/view?usp=share_link
python tools/test.py configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1Jj0Mdyd2sbaaredwNxVtp0zraHr_EgCN/view?usp=share_link
python tools/test.py configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1hf-Bod4nAo13dRgyHKYiAi260a1sYCT8/view?usp=share_link


# 多个GPU上测试
# 001 Classical Image Super-Resolution (middle size)
# (setting1: when model is trained on DIV2K and with training_patch_size=48)
./tools/dist_test.sh configs/swinir/swinir_x2s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1NwpDavsYKNcVptQyUrCzFp2ArAtBV6a2/view?usp=share_link
./tools/dist_test.sh configs/swinir/swinir_x3s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/11fn_CkgaYl-flzaeJeapKa0d17RwPNQ9/view?usp=share_link
./tools/dist_test.sh configs/swinir/swinir_x4s48w8d6e180_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1KWaJ3X6ZrXJZ37jHczdjRTcElQ_sPTpP/view?usp=share_link

# (setting2: when model is trained on DIV2K+Flickr2K and with training_patch_size=64)
./tools/dist_test.sh configs/swinir/swinir_x2s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/1Uw1QtPKnBvFalgx8sC-moFmTLCBtfbqM/view?usp=share_link
./tools/dist_test.sh configs/swinir/swinir_x3s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/13VOC_15bH4OcfqLX3TYa3NwoTSADKH9b/view?usp=share_link
./tools/dist_test.sh configs/swinir/swinir_x4s64w8d6e180_8xb4-lr2e-4-500k_df2k.py https://drive.google.com/file/d/12IXYTR_3UebYbTqR9wumBEv5AlL6Qyr9/view?usp=share_link

# 002 Lightweight Image Super-Resolution (small size)
./tools/dist_test.sh configs/swinir/swinir_x2s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/13dDwSMxjBpZZiXlgKHH9onkxzLUKZ0LX/view?usp=share_link
./tools/dist_test.sh configs/swinir/swinir_x3s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1Jj0Mdyd2sbaaredwNxVtp0zraHr_EgCN/view?usp=share_link
./tools/dist_test.sh configs/swinir/swinir_x4s64w8d4e60_8xb4-lr2e-4-500k_div2k.py https://drive.google.com/file/d/1hf-Bod4nAo13dRgyHKYiAi260a1sYCT8/view?usp=share_link

```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
