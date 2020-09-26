# Image Super-Resolution Using Deep Convolutional Networks

## Introduction

```
@article{dong2015image,
  title={Image super-resolution using deep convolutional networks},
  author={Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={38},
  number={2},
  pages={295--307},
  year={2015},
  publisher={IEEE}
}
```

## Results and Models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.

The metrics are `PSNR / SSIM`.

|            Method             |       Set5       |       Set14       |      DIV2K       |                                                                                                                    Download                                                                                                                     |
| :---------------------------: | :--------------: | :---------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| srcnn_x4k915_1x16_1000k_div2k | 28.4316 / 0.8099 | 25.6486 /  0.7014 | 27.7460 / 0.7854 | [model](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608_120159.log.json) |
