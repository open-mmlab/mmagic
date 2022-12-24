# RealBasicVSR (CVPR'2022)

> **任务**: 视频超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2111.12704">RealBasicVSR (CVPR'2022)</a></summary>

```bibtex
@InProceedings{chan2022investigating,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {RealBasicVSR: Investigating Tradeoffs in Real-World Video Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2022}
}
```

</details>

<br/>

在 Y 通道上评估。 计算 NRQM、NIQE 和 PI 的代码可以在[这里](https://github.com/roimehrez/PIRM2018)找到。我们使用 MATLAB 官方代码计算 BRISQUE。

|                                 算法                                 | NRQM (Y) | NIQE (Y) | PI (Y) | BRISQUE (Y) |         GPU 信息         |                                  Download                                   |
| :------------------------------------------------------------------: | :------: | :------: | :----: | :---------: | :----------------------: | :-------------------------------------------------------------------------: |
| [realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds](./realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py) |  6.0477  |  3.7662  | 3.8593 |   29.030    | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth)/[log](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104_183640.log.json) |
| [realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds](./realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py) |    -     |    -     |   -    |      -      | 8 (Tesla V100-SXM2-32GB) | [model](http://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds_20211027-0e2ff207.pth)/[log](http://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds_20211027_114039.log.json) |

## 训练

训练分为两个阶段：

1. 使用 [realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py](realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py) 训练一个没有感知损失和对抗性损失的模型。
2. 使用感知损失和对抗性损失 [realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py ](realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py) 微调模型。

**注:**

1. 您可能希望将图像裁剪为子图像以加快 IO。请参阅[此处](../../tools/dataset_converters/reds/preprocess_reds_dataset.py)了解更多详情。

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py

# 单个GPU上训练
python tools/train.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py

# 多个GPU上训练
./tools/dist_train.sh configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth

# 单个GPU上测试
python tools/test.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth

# 多个GPU上测试
./tools/dist_test.sh configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
