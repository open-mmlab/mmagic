# ESRGAN (ECCVW'2018)

<!-- [ALGORITHM] -->

<details>
<summary align="right">ESRGAN (ECCVW'2018)</summary>

```bibtex
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
  pages={0--0},
  year={2018}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                    算法                                     |       Set5        |      Set14       |      DIV2K       |                                     下载                                     |
| :-------------------------------------------------------------------------: | :---------------: | :--------------: | :--------------: | :--------------------------------------------------------------------------: |
| [esrgan_psnr_x4c64b23g32_1x16_1000k_div2k](/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py) | 30.6428 / 0.8559  | 27.0543 / 0.7447 | 29.3354 / 0.8263 | [模型](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420_112550.log.json) |
| [esrgan_x4c64b23g32_1x16_400k_div2k](/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py) | 28.2700 /  0.7778 | 24.6328 / 0.6491 | 26.6531 / 0.7340 | [模型](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508_191042.log.json) |
