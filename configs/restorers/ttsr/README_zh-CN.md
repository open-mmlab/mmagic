# TTSR (CVPR'2020)

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

|                                           算法                                            | scale |      CUFED       |                                            下载                                            |
| :---------------------------------------------------------------------------------------: | :---: | :--------------: | :----------------------------------------------------------------------------------------: |
| [ttsr-rec_x4_c64b16_g1_200k_CUFED](/configs/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED.py) |  x4   | 25.2433 / 0.7491 | [模型](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.log.json) |
| [ttsr-gan_x4_c64b16_g1_500k_CUFED](/configs/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED.py) |  x4   | 24.6075 / 0.7234 | [模型](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.log.json) |
