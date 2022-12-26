# SRCNN (TPAMI'2015)

> **任务**: 图像超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">SRCNN (TPAMI'2015)</summary>

```bibtex
@article{dong2015image,
  title={Image super-resolution using deep convolutional networks},
  author={Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={38},
  number={2},
  pages={295--307},
  year={2015},
  publisher={IEEE}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                 算法                                 |       Set5       |       Set14       |      DIV2K       | GPU 信息 |                                    下载                                     |
| :------------------------------------------------------------------: | :--------------: | :---------------: | :--------------: | :------: | :-------------------------------------------------------------------------: |
| [srcnn_x4k915_1x16_1000k_div2k](./srcnn_x4k915_1xb16-1000k_div2k.py) | 28.4316 / 0.8099 | 25.6486 /  0.7014 | 27.7460 / 0.7854 |    1     | [模型](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608_120159.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py

# 单个GPU上训练
python tools/train.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py

# 多个GPU上训练
./tools/dist_train.sh configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth

# 单个GPU上测试
python tools/test.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth

# 多个GPU上测试
./tools/dist_test.sh configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
