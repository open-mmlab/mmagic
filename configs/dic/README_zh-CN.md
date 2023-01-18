# DIC (CVPR'2020)

> **任务**: 图像超分辨率

<!-- [ALGORITHM] -->

<details>

<summary align="right">DIC (CVPR'2020)</summary>

```bibtex
@inproceedings{ma2020deep,
  title={Deep face super-resolution with iterative collaboration between attentive recovery and landmark estimation},
  author={Ma, Cheng and Jiang, Zhenyu and Rao, Yongming and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5569--5578},
  year={2020}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

在 `dic_gan_x8c48b6_g4_150k_CelebAHQ` 的日志中，DICGAN 在 CelebA-HQ 测试集的前9张图片上进行了验证，因此下表中的 `PSNR/SSIM` 与日志数据不同。

`GPU 信息`: 训练过程中的 GPU 信息.

|                                     算法                                     | scale |    CelebA-HQ     |      GPU 信息       |                                         下载                                         |
| :--------------------------------------------------------------------------: | :---: | :--------------: | :-----------------: | :----------------------------------------------------------------------------------: |
|     [dic_x8c48b6_g4_150k_CelebAHQ](./dic_x8c48b6_4xb2-150k_celeba-hq.py)     |  x8   | 25.2319 / 0.7422 | 4 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.log.json) |
| [dic_gan_x8c48b6_g4_500k_CelebAHQ](./dic_gan-x8c48b6_4xb2-500k_celeba-hq.py) |  x8   | 23.6241 / 0.6721 | 4 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py

# 单个GPU上训练
python tools/train.py configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py

# 多个GPU上训练
./tools/dist_train.sh configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth

# 单个GPU上测试
python tools/test.py configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth

# 多个GPU上测试
./tools/dist_test.sh configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
