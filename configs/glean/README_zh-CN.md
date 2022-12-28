# GLEAN (CVPR'2021)

> **任务**: 图像超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">GLEAN (CVPR'2021)</summary>

```bibtex
@InProceedings{chan2021glean,
  author = {Chan, Kelvin CK and Wang, Xintao and Xu, Xiangyu and Gu, Jinwei and Loy, Chen Change},
  title = {GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

</details>

<br/>

有关训练和测试中使用的元信息，请参阅[此处](https://github.com/ckkelvinchan/GLEAN)。 结果在 RGB 通道上进行评估。

|                                         算法                                          | PSNR  |         GPU 信息         |                                          下载                                          |
| :-----------------------------------------------------------------------------------: | :---: | :----------------------: | :------------------------------------------------------------------------------------: |
|                        [glean_cat_8x](./glean_x8_2xb8_cat.py)                         | 23.98 | 2 (Tesla V100-PCIE-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614-d3ac8683.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614_145540.log.json) |
|                      [glean_ffhq_16x](./glean_x16_2xb8_ffhq.py)                       | 26.91 | 2 (Tesla V100-PCIE-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527-61a3afad.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527_194536.log.json) |
|                       [glean_cat_16x](./glean_x16_2xb8_cat.py)                        | 20.88 | 2 (Tesla V100-PCIE-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527-68912543.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527_103708.log.json) |
| [glean_in128out1024_4x2_300k_ffhq_celebahq](./glean_in128out1024_4xb2-300k_ffhq-celeba-hq.py) | 27.94 | 4 (Tesla V100-SXM3-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812_100549.log.json) |
|                   [glean_fp16_cat_8x](./glean_x8-fp16_2xb8_cat.py)                    |   -   |            -             |                                           -                                            |
|                 [glean_fp16_ffhq_16x](./glean_x16-fp16_2xb8_ffhq.py)                  |   -   |            -             |                                           -                                            |
| [glean_fp16_in128out1024_4x2_300k_ffhq_celebahq](./glean_in128out1024-fp16_4xb2-300k_ffhq-celeba-hq.py) |   -   |            -             |                                           -                                            |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/glean/glean_x8_2xb8_cat.py

# 单个GPU上训练
python tools/train.py configs/glean/glean_x8_2xb8_cat.py

# 多个GPU上训练
./tools/dist_train.sh configs/glean/glean_x8_2xb8_cat.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/glean/glean_x8_2xb8_cat.py https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614-d3ac8683.pth

# 单个GPU上测试
python tools/test.py configs/glean/glean_x8_2xb8_cat.py https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614-d3ac8683.pth

# 多个GPU上测试
./tools/dist_test.sh configs/glean/glean_x8_2xb8_cat.py https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614-d3ac8683.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
