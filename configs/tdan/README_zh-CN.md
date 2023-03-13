# TDAN (CVPR'2020)

> **任务**: 视频超分辨率

<!-- [ALGORITHM] -->

<details>
<summary align="right">TDAN (CVPR'2020)</summary>

```bibtex
@InProceedings{tian2020tdan,
  title={TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution},
  author={Tian, Yapeng and Zhang, Yulun and Fu, Yun and Xu, Chenliang},
  booktitle = {Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的8像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                            算法                            |   Vid4 (BIx4)   | SPMCS-30 (BIx4) |   Vid4 (BDx4)   | SPMCS-30 (BDx4) |         GPU 信息         |                            下载                            |
| :--------------------------------------------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :----------------------: | :--------------------------------------------------------: |
| [tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi](./tdan_x4_8xb16-lr1e-4-400k_vimeo90k-bi.py) |        -        |        -        |        -        |        -        | 8 (Tesla V100-SXM2-32GB) |                             -                              |
| [tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bd](./tdan_x4_8xb16-lr1e-4-400k_vimeo90k-bd.py) |        -        |        -        |        -        |        -        | 8 (Tesla V100-SXM2-32GB) |                             -                              |
| [tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi](./tdan_x4ft_8xb16-lr5e-5-400k_vimeo90k-bi.py) | **26.49/0.792** | **30.42/0.856** |   25.93/0.772   |   29.69/0.842   | 8 (Tesla V100-SXM2-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_x4ft_1xb16-lr5e-5-800k_vimeo90k-bd](./tdan_x4ft_8xb16-lr5e-5-800k_vimeo90k-bd.py) |   25.80/0.784   |   29.56/0.851   | **26.87/0.815** | **30.77/0.868** | 8 (Tesla V100-SXM2-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

TDAN 训练有两个阶段。

**阶段 1**: 以更大的学习率训练 (1e-4)

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/tdan/tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi.py

# 单个GPU上训练
python tools/train.py configs/tdan/tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi.py

# 多个GPU上训练
./tools/dist_train.sh configs/tdan/tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi.py 8
```

**阶段 2**: 以较小的学习率进行微调 (5e-5)

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py

# 单个GPU上训练
python tools/train.py configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py

# 多个GPU上训练
./tools/dist_train.sh configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth

# 单个GPU上测试
python tools/test.py configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth

# 多个GPU上测试
./tools/dist_test.sh configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
