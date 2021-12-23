# EDVR (CVPRW'2019)

<!-- [ALGORITHM] -->

<details>
<summary align="right">EDVR (CVPRW'2019)</summary>

```bibtex
@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```

</details>

<br/>

在 RGB 通道上进行评估。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                        算法                                         |       REDS4       |                                                                                                                  下载                                                                                                                   |
| :-----------------------------------------------------------------------------------: | :---------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [edvrm_wotsa_x4_8x4_600k_reds](/configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py) | 30.3430 /  0.8664 | [模型](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522_141644.log.json) |
|       [edvrm_x4_8x4_600k_reds](/configs/restorers/edvr/edvrm_x4_g8_600k_reds.py)       | 30.4194 / 0.8684  |       [模型](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20200622_102544.log.json)       |
