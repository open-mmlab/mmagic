# RDN (CVPR'2018)

> **任务**: 图像超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">RDN (CVPR'2018)</summary>

```bibtex
@inproceedings{zhang2018residual,
  title={Residual dense network for image super-resolution},
  author={Zhang, Yulun and Tian, Yapeng and Kong, Yu and Zhong, Bineng and Fu, Yun},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2472--2481},
  year={2018}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                算法                                |       Set5       |      Set14       |      DIV2K       |   GPU 信息   |                                    下载                                    |
| :----------------------------------------------------------------: | :--------------: | :--------------: | :--------------: | :----------: | :------------------------------------------------------------------------: |
| [rdn_x2c64b16_g1_1000k_div2k](./rdn_x2c64b16_1xb16-1000k_div2k.py) | 35.9883 / 0.9385 | 31.8366 / 0.8920 | 34.9392 / 0.9380 | 1 (TITAN Xp) | [模型](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_20210419-dc146009.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_20210419-dc146009.log.json) |
| [rdn_x3c64b16_g1_1000k_div2k](./rdn_x3c64b16_1xb16-1000k_div2k.py) | 32.6051 / 0.8943 | 28.6338 / 0.8077 | 31.2153 / 0.8763 | 1 (TITAN Xp) | [模型](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x3c64b16_g1_1000k_div2k_20210419-b93cb6aa.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x3c64b16_g1_1000k_div2k_20210419-b93cb6aa.log.json) |
| [rdn_x4c64b16_g1_1000k_div2k](./rdn_x4c64b16_1xb16-1000k_div2k.py) | 30.4922 / 0.8548 | 26.9570 / 0.7423 | 29.1925 / 0.8233 | 1 (TITAN Xp) | [模型](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/rdn/rdn_x4c64b16_1xb16-1000k_div2k.py

# 单个GPU上训练
python tools/train.py configs/rdn/rdn_x4c64b16_1xb16-1000k_div2k.py

# 多个GPU上训练
./tools/dist_train.sh configs/rdn/rdn_x4c64b16_1xb16-1000k_div2k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/rdn/rdn_x4c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.pth

# 单个GPU上测试
python tools/test.py configs/rdn/rdn_x4c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.pth

# 多个GPU上测试
./tools/dist_test.sh configs/rdn/rdn_x4c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
