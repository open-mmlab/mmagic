# TDAN (CVPR'2020)

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

|                                  算法                                  |   Vid4 (BIx4)   | SPMCS-30 (BIx4) |   Vid4 (BDx4)   | SPMCS-30 (BDx4) |                                  下载                                  |
| :--------------------------------------------------------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :--------------------------------------------------------------------: |
| [tdan_vimeo90k_bix4_ft_lr5e-5_400k](/configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py) | **26.49/0.792** | **30.42/0.856** |   25.93/0.772   |   29.69/0.842   | [模型](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_vimeo90k_bdx4_ft_lr5e-5_800k](/configs/restorers/tdan/tdan_vimeo90k_bdx4_ft_lr5e-5_800k.py) |   25.80/0.784   |   29.56/0.851   | **26.87/0.815** | **30.77/0.868** | [模型](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

TDAN 训练有两个阶段。

**阶段 1**: 以更大的学习率训练 (1e-4)

```shell
./tools/dist_train.sh configs/restorers/tdan/tdan_vimeo90k_bix4_lr1e-4_400k.py 8
```

**阶段 2**: 以较小的学习率进行微调 (5e-5)

```shell
./tools/dist_train.sh configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py 8
```

更多细节可以参考 [getting_started](/docs/zh_cn/getting_started.md#train-a-model) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```

示例：使用 `bicubic` 下采样在 SPMCS-30 上测试 TDAN。

```shell
python tools/test.py configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py  checkpoints/SOME_CHECKPOINT.pth --save_path outputs/
```

更多细节可以参考 [getting_started](/docs/zh_cn/getting_started.md#inference-with-pretrained-models) 中的 **Inference with pretrained models** 部分。

</details>
