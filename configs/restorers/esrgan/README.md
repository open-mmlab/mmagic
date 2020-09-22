# ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks

## Introduction

```
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
  pages={0--0},
  year={2018}
}
```

## Results and Models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.

The metrics are `PSNR / SSIM`.

|                  Method                  |       Set5        |      Set14       |      DIV2K       |                                                                                                                                Download                                                                                                                                 |
| :--------------------------------------: | :---------------: | :--------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| esrgan_psnr_x4c64b23g32_1x16_1000k_div2k | 30.6428 / 0.8559  | 27.0543 / 0.7447 | 29.3354 / 0.8263 | [model](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420_112550.log.json) |
|    esrgan_x4c64b23g32_1x16_400k_div2k    | 28.2700 /  0.7778 | 24.6328 / 0.6491 | 26.6531 / 0.7340 |       [model](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508_191042.log.json)       |
