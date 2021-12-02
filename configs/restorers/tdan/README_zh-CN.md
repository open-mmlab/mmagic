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

Evaluated on Y-channel. 8 pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                               Method                                              |   Vid4 (BIx4)   | SPMCS-30 (BIx4) |   Vid4 (BDx4)   | SPMCS-30 (BDx4) |                                                                                                         Download                                                                                                        |
|:-------------------------------------------------------------------------------------------------:|:---------------:|:---------------:|:---------------:|:---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [tdan_vimeo90k_bix4_ft_lr5e-5_400k](/configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py) | **26.49/0.792** | **30.42/0.856** |   25.93/0.772   |   29.69/0.842   | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_vimeo90k_bdx4_ft_lr5e-5_800k](/configs/restorers/tdan/tdan_vimeo90k_bdx4_ft_lr5e-5_800k.py) |   25.80/0.784   |   29.56/0.851   | **26.87/0.815** | **30.77/0.868** | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following command to train a model.

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

TDAN is trained with two stages.

**Stage 1**: Train with a larger learning rate (1e-4)

```shell
./tools/dist_train.sh configs/restorers/tdan/tdan_vimeo90k_bix4_lr1e-4_400k.py 8
```

**Stage 2**: Fine-tune with a smaller learning rate (5e-5)

```shell
./tools/dist_train.sh configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py 8
```

For more details, you can refer to **Train a model** part in [getting_started](/docs/getting_started.md#train-a-model).
</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```

Example: Test TDAN on SPMCS-30 using Bicubic downsampling.

```shell
python tools/test.py configs/restorers/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py  checkpoints/SOME_CHECKPOINT.pth --save_path outputs/
```

For more details, you can refer to **Inference with pretrained models** part in [getting_started](/docs/getting_started.md#inference-with-pretrained-models).
</details>
