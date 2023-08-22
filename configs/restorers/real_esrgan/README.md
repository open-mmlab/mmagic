# Real-ESRGAN (ICCVW'2021)

> [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Though many attempts have been made in blind super-resolution to restore low-resolution images with unknown and complex degradations, they are still far from addressing general real-world degraded images. In this work, we extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data. Specifically, a high-order degradation modeling process is introduced to better simulate complex real-world degradations. We also consider the common ringing and overshoot artifacts in the synthesis process. In addition, we employ a U-Net discriminator with spectral normalization to increase discriminator capability and stabilize the training dynamics. Extensive comparisons have shown its superior visual performance than prior works on various real datasets. We also provide efficient implementations to synthesize training pairs on the fly.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144034533-f81430df-351b-490c-9e00-733465edf3ee.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels. The metrics are `PSNR/SSIM`.

|                                            Method                                            |      Set5      |                                            Download                                            |
| :------------------------------------------------------------------------------------------: | :------------: | :--------------------------------------------------------------------------------------------: |
| [realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost](/configs/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost.py) | 28.0297/0.8236 | [model](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost_20210816-4ae3b5a4.pth)/log |
| [realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost](/configs/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost.py) | 26.2204/0.7655 | [model](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20211010-34798885.pth) /[log](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20210922_142838.log.json) |

## Citation

```bibtex
@inproceedings{wang2021real,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={1905--1914},
  year={2021}
}
```
