# BasicVSR++ (CVPR'2022)

> [BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment](https://arxiv.org/abs/2104.13371)

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

|     Method      | REDS4 (BIx4) PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4) PSNR/SSIM (Y) | Vid4 (BIx4) PSNR/SSIM (Y) | UDM10 (BDx4) PSNR/SSIM (Y) | Vimeo-90K-T (BDx4) PSNR/SSIM (Y) | Vid4 (BDx4) PSNR/SSIM (Y) |     Download      |
| :-------------: | :--------------------------: | :------------------------------: | :-----------------------: | :------------------------: | :------------------------------: | :-----------------------: | :---------------: |
| [basicvsr_plusplus_c64n7_8x1_600k_reds4](/configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py) |      **32.3855/0.9069**      |          36.4445/0.9411          |      27.7674/0.8444       |       34.6868/0.9417       |          34.0372/0.9244          |      24.6209/0.7540       | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217_113115.log.json) |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi](/configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi.py) |        31.0126/0.8804        |        **37.7864/0.9500**        |    **27.7882/0.8401**     |       33.1211/0.9270       |          33.8972/0.9195          |      23.6086/0.7033       | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305_141254.log.json) |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd](/configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd.py) |        29.2041/0.8528        |          34.7248/0.9351          |      26.4377/0.8074       |     **40.7216/0.9722**     |        **38.2054/0.9550**        |    **29.0400/0.8753**     | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305_140921.log.json) |

<details>
<summary align="left">NTIRE 2021 checkpoints</summary>

Note that the following models are finetuned from smaller models. The training schemes of these models will be released when MMEditing reaches 5k stars. We provide the pre-trained models here.

[NTIRE 2021 Video Super-Resolution](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 1](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 2](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 3](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40.pth)

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
