# EDVR (CVPRW'2019)

> [EDVR: Video Restoration with Enhanced Deformable Convolutional Networks](https://arxiv.org/abs/1905.02716?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video restoration tasks, including super-resolution, deblurring, etc, are drawing increasing attention in the computer vision community. A challenging benchmark named REDS is released in the NTIRE19 Challenge. This new benchmark challenges existing methods from two aspects: (1) how to align multiple frames given large motions, and (2) how to effectively fuse different frames with diverse motion and blur. In this work, we propose a novel Video Restoration framework with Enhanced Deformable networks, termed EDVR, to address these challenges. First, to handle large motions, we devise a Pyramid, Cascading and Deformable (PCD) alignment module, in which frame alignment is done at the feature level using deformable convolutions in a coarse-to-fine manner. Second, we propose a Temporal and Spatial Attention (TSA) fusion module, in which attention is applied both temporally and spatially, so as to emphasize important features for subsequent restoration. Thanks to these modules, our EDVR wins the champions and outperforms the second place by a large margin in all four tracks in the NTIRE19 video restoration and enhancement challenges. EDVR also demonstrates superior performance to state-of-the-art published methods on video super-resolution and deblurring.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144018263-6a1f74a4-d011-47fd-906b-290dd77eed64.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                           Method                                            |      REDS4       |                                           Download                                            |
| :-----------------------------------------------------------------------------------------: | :--------------: | :-------------------------------------------------------------------------------------------: |
|   [edvrm_wotsa_x4_8x4_600k_reds](/configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py)    | 30.3430 / 0.8664 | [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522_141644.log.json) |
|         [edvrm_x4_8x4_600k_reds](/configs/restorers/edvr/edvrm_x4_g8_600k_reds.py)          | 30.4194 / 0.8684 | [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20200622_102544.log.json) |
| [edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4](/configs/restorers/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4.py) | 31.0010 / 0.8784 | [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4_20211228-d895a769.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4_20211228_144658.log.json) |
| [edvrl_c128b40_8x8_lr2e-4_600k_reds4](/configs/restorers/edvr/edvrl_c128b40_8x8_lr2e-4_600k_reds4.py) | 31.0467 / 0.8793 | [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104-4509865f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104_171823.log.json) |

## Citation

```bibtex
@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```
