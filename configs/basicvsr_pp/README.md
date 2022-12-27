# BasicVSR++ (CVPR'2022)

> [BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment](https://arxiv.org/abs/2104.13371)

> **Task**: Video Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

A recurrent structure is a popular framework choice for the task of video super-resolution. The state-of-the-art method BasicVSR adopts bidirectional propagation with feature alignment to effectively exploit information from the entire input video. In this study, we redesign BasicVSR by proposing second-order grid propagation and flow-guided deformable alignment. We show that by empowering the recurrent framework with the enhanced propagation and alignment, one can exploit spatiotemporal information across misaligned video frames more effectively. The new components lead to an improved performance under a similar computational constraint. In particular, our model BasicVSR++ surpasses BasicVSR by 0.82 dB in PSNR with similar number of parameters. In addition to video super-resolution, BasicVSR++ generalizes well to other video restoration tasks such as compressed video enhancement. In NTIRE 2021, BasicVSR++ obtains three champions and one runner-up in the Video Super-Resolution and Compressed Video Enhancement Challenges. Codes and models will be released to MMEditing.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144017685-9354df55-aa6d-445f-a946-116f0d6c38d7.png" width="400"/>
</div >

## Results and models

The pretrained weights of SPyNet can be found [here](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth).

|       Method        | REDS4 (BIx4) PSNR (RGB) | Vimeo-90K-T (BIx4) PSNR (Y) | Vid4 (BIx4) PSNR (Y) | UDM10 (BDx4) PSNR (Y) | Vimeo-90K-T (BDx4) PSNR (Y) | Vid4 (BDx4) PSNR (Y) |       GPU Info        |        Download        |
| :-----------------: | :---------------------: | :-------------------------: | :------------------: | :-------------------: | :-------------------------: | :------------------: | :-------------------: | :--------------------: |
| [basicvsr_plusplus_c64n7_8x1_600k_reds4](./basicvsr-pp_c64n7_8xb1-600k_reds4.py) |       **32.3855**       |           36.4445           |       27.7674        |        34.6868        |           34.0372           |       24.6209        | 8 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217_113115.log.json) |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi](./basicvsr-pp_c64n7_4xb2-300k_vimeo90k-bi.py) |         31.0126         |         **37.7864**         |     **27.7882**      |        33.1211        |           33.8972           |       23.6086        | 4 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305_141254.log.json) |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd](./basicvsr-pp_c64n7_4xb2-300k_vimeo90k-bd.py) |         29.2041         |           34.7248           |       26.4377        |      **40.7216**      |         **38.2054**         |     **29.0400**      | 4 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305_140921.log.json) |

|       Method        | REDS4 (BIx4) SSIM (RGB) | Vimeo-90K-T (BIx4) SSIM (Y) | Vid4 (BIx4) SSIM (Y) | UDM10 (BDx4) SSIM (Y) | Vimeo-90K-T (BDx4) SSIM (Y) | Vid4 (BDx4) SSIM (Y) |       GPU Info        |        Download        |
| :-----------------: | :---------------------: | :-------------------------: | :------------------: | :-------------------: | :-------------------------: | :------------------: | :-------------------: | :--------------------: |
| [basicvsr_plusplus_c64n7_8x1_600k_reds4](./basicvsr-pp_c64n7_8xb1-600k_reds4.py) |       **0.9069**        |           0.9411            |        0.8444        |        0.9417         |           0.9244            |        0.7540        | 8 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217_113115.log.json) |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi](./basicvsr-pp_c64n7_4xb2-300k_vimeo90k-bi.py) |         0.8804          |         **0.9500**          |      **0.8401**      |        0.9270         |           0.9195            |        0.7033        | 4 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305_141254.log.json) |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd](./basicvsr-pp_c64n7_4xb2-300k_vimeo90k-bd.py) |         0.8528          |           0.9351            |        0.8074        |      **0.9722**       |         **0.9550**          |      **0.8753**      | 4 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305_140921.log.json) |

<details>
<summary align="left">NTIRE 2021 checkpoints</summary>

Note that the following models are finetuned from smaller models. The training schemes of these models will be released when MMEditing reaches 5k stars. We provide the pre-trained models here.

| Method                                                                | Download                                                                | Track                                                        |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------ |
| [basicvsr-pp_c128n25_600k_ntire-vsr](./basicvsr-pp_c128n25_600k_ntire-vsr.py) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth) | NTIRE 2021 Video Super-Resolution                            |
| [basicvsr-pp_c128n25_600k_ntire-decompress-track1](./basicvsr-pp_c128n25_600k_ntire-decompress-track1.py) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth) | NTIRE 2021 Quality Enhancement of Compressed Video - Track 1 |
| [basicvsr-pp_c128n25_600k_ntire-decompress-track2](./basicvsr-pp_c128n25_600k_ntire-decompress-track2.py) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6.pth) | NTIRE 2021 Quality Enhancement of Compressed Video - Track 2 |
| [basicvsr-pp_c128n25_600k_ntire-decompress-track3](./basicvsr-pp_c128n25_600k_ntire-decompress-track3.py) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40.pth) | NTIRE 2021 Quality Enhancement of Compressed Video - Track 3 |

</details>

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py

# single-gpu train
python tools/train.py configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py

# multi-gpu train
./tools/dist_train.sh configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

# single-gpu test
python tools/test.py configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

# multi-gpu test
./tools/dist_test.sh configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>

## Citation

```bibtex
@InProceedings{chan2022basicvsrplusplus,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2022}
}
```
