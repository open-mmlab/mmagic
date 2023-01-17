# EDSR (CVPR'2017)

> **任务**: 图像超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">EDSR (CVPR'2017)</summary>

```bibtex
@inproceedings{lim2017enhanced,
  title={Enhanced deep residual networks for single image super-resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={136--144},
  year={2017}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                 算法                                 |       Set5       |      Set14       |      DIV2K       | GPU 信息 |                                     下载                                     |
| :------------------------------------------------------------------: | :--------------: | :--------------: | :--------------: | :------: | :--------------------------------------------------------------------------: |
| [edsr_x2c64b16_1x16_300k_div2k](./edsr_x2c64b16_1xb16-300k_div2k.py) | 35.7592 / 0.9372 | 31.4290 / 0.8874 | 34.5896 / 0.9352 |    1     | [模型](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x2c64b16_1x16_300k_div2k_20200604-19fe95ea.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x2c64b16_1x16_300k_div2k_20200604_221933.log.json) |
| [edsr_x3c64b16_1x16_300k_div2k](./edsr_x3c64b16_1xb16-300k_div2k.py) | 32.3301 / 0.8912 | 28.4125 / 0.8022 | 30.9154 / 0.8711 |    1     | [模型](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x3c64b16_1x16_300k_div2k_20200608-36d896f4.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x3c64b16_1x16_300k_div2k_20200608_114850.log.json) |
| [edsr_x4c64b16_1x16_300k_div2k](./edsr_x4c64b16_1xb16-300k_div2k.py) | 30.2223 / 0.8500 | 26.7870 / 0.7366 | 28.9675 / 0.8172 |    1     | [模型](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608_115148.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py

# 单个GPU上训练
python tools/train.py configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py

# 多个GPU上训练
./tools/dist_train.sh configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth

# 单个GPU上测试
python tools/test.py configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth

# 多个GPU上测试
./tools/dist_test.sh configs/edsr/edsr_x4c64b16_1xb16-300k_div2k.py https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
