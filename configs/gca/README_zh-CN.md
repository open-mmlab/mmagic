# GCA (AAAI'2020)

> **任务**: 图像抠图

<!-- [ALGORITHM] -->

<details>
<summary align="right">GCA (AAAI'2020)</summary>

```bibtex
@inproceedings{li2020natural,
  title={Natural Image Matting via Guided Contextual Attention},
  author={Li, Yaoyi and Lu, Hongtao},
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2020}
}
```

</details>

<br/>

|                        算法                        |    SAD    |    MSE     |   GRAD    |   CONN    | GPU 信息 |                                                   下载                                                    |
| :------------------------------------------------: | :-------: | :--------: | :-------: | :-------: | :------: | :-------------------------------------------------------------------------------------------------------: |
|                    基线 (原文)                     |   40.62   |   0.0106   |   21.53   |   38.43   |    -     |                                                     -                                                     |
|                     GCA (原文)                     |   35.28   |   0.0091   |   16.92   |   32.53   |    -     |                                                     -                                                     |
| [基线 (复现)](./baseline_r34_4xb10-200k_comp1k.py) |   34.61   |   0.0083   |   16.21   |   32.12   |    4     | [模型](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-34.61_20220620-96f85d56.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-34.61_20220620-96f85d56.log) |
|    [GCA (复现)](./gca_r34_4xb10-200k_comp1k.py)    | **33.38** | **0.0081** | **14.96** | **30.59** |    4     | [模型](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.log) |

**其他结果**

|                                 算法                                 |  SAD  |  MSE   | GRAD  | CONN  | GPU 信息 |                                                  下载                                                   |
| :------------------------------------------------------------------: | :---: | :----: | :---: | :---: | :------: | :-----------------------------------------------------------------------------------------------------: |
| [基线 (使用 DIM 流水线)](./baseline_r34_4xb10-dimaug-200k_comp1k.py) | 49.95 | 0.0144 | 30.21 | 49.67 |    4     | [模型](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_SAD-49.95_20200626_231612-535c9a11.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_20200626_231612.log.json) |
|    [GCA (使用 DIM 流水线)](./gca_r34_4xb10-dimaug-200k_comp1k.py)    | 49.42 | 0.0129 | 28.07 | 49.47 |    4     | [模型](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_SAD-49.42_20200626_231422-8e9cc127.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_20200626_231422.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/gca/gca_r34_4xb10-200k_comp1k.py

# 单个GPU上训练
python tools/train.py configs/gca/gca_r34_4xb10-200k_comp1k.py

# 多个GPU上训练
./tools/dist_train.sh configs/gca/gca_r34_4xb10-200k_comp1k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/gca/gca_r34_4xb10-200k_comp1k.py https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth

# 单个GPU上测试
python tools/test.py configs/gca/gca_r34_4xb10-200k_comp1k.py https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth

# 多个GPU上测试
./tools/dist_test.sh configs/gca/gca_r34_4xb10-200k_comp1k.py https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
