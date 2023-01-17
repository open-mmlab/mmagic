# EDVR (CVPRW'2019)

> **任务**: 视频超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">EDVR (CVPRW'2019)</summary>

```bibtex
@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```

</details>

<br/>

在 RGB 通道上进行评估。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                       算法                                       |      REDS4       |         GPU 信息         |                                       下载                                       |
| :------------------------------------------------------------------------------: | :--------------: | :----------------------: | :------------------------------------------------------------------------------: |
|         [edvrm_wotsa_x4_8x4_600k_reds](./edvrm_wotsa_8xb4-600k_reds.py)          | 30.3430 / 0.8664 |            8             | [模型](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522_141644.log.json) |
|               [edvrm_x4_8x4_600k_reds](./edvrm_8xb4-600k_reds.py)                | 30.4194 / 0.8684 |            8             | [模型](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20200622_102544.log.json) |
| [edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4](./edvrl_wotsa-c128b40_8xb8-lr2e-4-600k_reds4.py) | 31.0010 / 0.8784 | 8 (Tesla V100-PCIE-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4_20211228-d895a769.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4_20211228_144658.log.json) |
| [edvrl_c128b40_8x8_lr2e-4_600k_reds4](./edvrl_c128b40_8xb8-lr2e-4-600k_reds4.py) | 31.0467 / 0.8793 | 8 (Tesla V100-PCIE-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104-4509865f.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104_171823.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/edvr/edvrm_8xb4-600k_reds.py

# 单个GPU上训练
python tools/train.py configs/edvr/edvrm_8xb4-600k_reds.py

# 多个GPU上训练
./tools/dist_train.sh configs/edvr/edvrm_8xb4-600k_reds.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/edvr/edvrm_8xb4-600k_reds.py https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth

# 单个GPU上测试
python tools/test.py configs/edvr/edvrm_8xb4-600k_reds.py https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth

# 多个GPU上测试
./tools/dist_test.sh configs/edvr/edvrm_8xb4-600k_reds.py https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
