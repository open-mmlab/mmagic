# SRGAN (CVPR'2016)

> **任务**: 图像超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">SRGAN (CVPR'2016)</summary>

```bibtex
@inproceedings{ledig2016photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  year={2016}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                  算法                                   |       Set5        |      Set14       |      DIV2K       | GPU 信息 |                                   下载                                   |
| :---------------------------------------------------------------------: | :---------------: | :--------------: | :--------------: | :------: | :----------------------------------------------------------------------: |
| [msrresnet_x4c64b16_1x16_300k_div2k](./msrresnet_x4c64b16_1xb16-1000k_div2k.py) | 30.2252 / 0.8491  | 26.7762 / 0.7369 | 28.9748 / 0.8178 |    1     | [模型](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521-61556be5.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521_110246.log.json) |
| [srgan_x4c64b16_1x16_1000k_div2k](./srgan_x4c64b16_1xb16-1000k_div2k.py) | 27.9499 /  0.7846 | 24.7383 / 0.6491 | 26.5697 / 0.7365 |    1     | [模型](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200606-a1f0810e.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200506_191442.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/srgan_resnet/srgan_x4c64b16_1xb16-1000k_div2k.py

# 单个GPU上训练
python tools/train.py configs/srgan_resnet/srgan_x4c64b16_1xb16-1000k_div2k.py

# 多个GPU上训练
./tools/dist_train.sh configs/srgan_resnet/srgan_x4c64b16_1xb16-1000k_div2k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/srgan_resnet/srgan_x4c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200606-a1f0810e.pth

# 单个GPU上测试
python tools/test.py configs/srgan_resnet/srgan_x4c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200606-a1f0810e.pth

# 多个GPU上测试
./tools/dist_test.sh configs/srgan_resnet/srgan_x4c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200606-a1f0810e.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
