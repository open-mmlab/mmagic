# BasicVSR++ (CVPR'2022)

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2104.13371">BasicVSR++ (CVPR'2022)</a></summary>

```bibtex
@InProceedings{chan2022basicvsrplusplus,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2022}
}
```

```

</details>

SPyNet 的 预训练权重在[这里](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth)。

|                                                 算法                                                                      | REDS4 (BIx4) PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4) PSNR/SSIM (Y) | Vid4 (BIx4) PSNR/SSIM (Y) | UDM10 (BDx4) PSNR/SSIM (Y) | Vimeo-90K-T (BDx4) PSNR/SSIM (Y) | Vid4 (BDx4) PSNR/SSIM (Y) |                                                                                                                                                Download                                                                                                                                               |
| :-------------------------------------------------------------------------------------------------------------------------: |:----------------------------:|:--------------------------------:|:-------------------------:|:--------------------------:|:--------------------------------:|:-------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [basicvsr_plusplus_c64n7_8x1_600k_reds4](/configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py)             |      **32.3855/0.9069**      |          36.4445/0.9411          |       27.7674/0.8444      |       34.6868/0.9417       |          34.0372/0.9244          |       24.6209/0.7540      |       [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217_113115.log.json)       |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi](/configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi.py) |        31.0126/0.8804        |        **37.7864/0.9500**        |     **27.7882/0.8401**    |       33.1211/0.9270       |          33.8972/0.9195          |       23.6086/0.7033      | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305_141254.log.json) |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd](/configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd.py) |        29.2041/0.8528        |          34.7248/0.9351          |       26.4377/0.8074      |     **40.7216/0.9722**     |        **38.2054/0.9550**        |     **29.0400/0.8753**    | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305_140921.log.json) |

<details>
<summary align="left">NTIRE 2021 模型权重文件</summary>

请注意，以下模型是从较小的模型中微调而来的。 这些模型的训练方案将在 MMEditing 达到 5k star 时发布。 我们在这里提供预训练的模型。

[NTIRE 2021 Video Super-Resolution](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 1](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 2](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 3](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40.pth)

</details>
```
