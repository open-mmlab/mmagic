# TTSR (CVPR'2020)

> **任务**: 图像超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">TTSR (CVPR'2020)</summary>

```bibtex
@inproceedings{yang2020learning,
  title={Learning texture transformer network for image super-resolution},
  author={Yang, Fuzhi and Yang, Huan and Fu, Jianlong and Lu, Hongtao and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5791--5800},
  year={2020}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                    算法                                    | scale |      CUFED       |   GPU 信息   |                                             下载                                              |
| :------------------------------------------------------------------------: | :---: | :--------------: | :----------: | :-------------------------------------------------------------------------------------------: |
| [ttsr-rec_x4_c64b16_g1_200k_CUFED](./ttsr-rec_x4c64b16_1xb9-200k_CUFED.py) |  x4   | 25.2433 / 0.7491 | 1 (TITAN Xp) | [模型](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.log.json) |
| [ttsr-gan_x4_c64b16_g1_500k_CUFED](./ttsr-gan_x4c64b16_1xb9-500k_CUFED.py) |  x4   | 24.6075 / 0.7234 | 1 (TITAN Xp) | [模型](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py

# 单个GPU上训练
python tools/train.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py

# 多个GPU上训练
./tools/dist_train.sh configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth

# 单个GPU上测试
python tools/test.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth

# 多个GPU上测试
./tools/dist_test.sh configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
