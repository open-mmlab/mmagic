# ESRGAN (ECCVW'2018)

> **任务**: 图像超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">ESRGAN (ECCVW'2018)</summary>

```bibtex
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
  pages={0--0},
  year={2018}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                  算法                                   |       Set5        |      Set14       |      DIV2K       | GPU 信息 |                                   下载                                   |
| :---------------------------------------------------------------------: | :---------------: | :--------------: | :--------------: | :------: | :----------------------------------------------------------------------: |
| [esrgan_psnr_x4c64b23g32_1x16_1000k_div2k](./esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py) | 30.6428 / 0.8559  | 27.0543 / 0.7447 | 29.3354 / 0.8263 |    1     | [模型](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420_112550.log.json) |
| [esrgan_x4c64b23g32_1x16_400k_div2k](./esrgan_x4c64b23g32_1xb16-400k_div2k.py) | 28.2700 /  0.7778 | 24.6328 / 0.6491 | 26.6531 / 0.7340 |    1     | [模型](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508_191042.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py

# 单个GPU上训练
python tools/train.py configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py

# 多个GPU上训练
./tools/dist_train.sh configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth

# 单个GPU上测试
python tools/test.py configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth

# 多个GPU上测试
./tools/dist_test.sh configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
