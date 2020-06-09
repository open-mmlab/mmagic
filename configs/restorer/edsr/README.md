# EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution

## Introduction

```
@inproceedings{lim2017enhanced,
  title={Enhanced deep residual networks for single image super-resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={136--144},
  year={2017}
}
```

## Results and Models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.

The metrics are `PSNR / SSIM`.

|   Method   |  Set5  | Set14 | DIV2K | Download |
|:----------:|:----:|:-----:|:----:|:--------:|
| edsr_x2c64b16_g1_300k_div2k | 35.7592 / 0.9372 | 31.4290 / 0.8874 | 34.5896 / 0.9352 | [model](TODO) &#124; [log](TODO) |
| edsr_x3c64b16_g1_300k_div2k | 32.3301 / 0.8912 | 28.4125 / 0.8022 | 30.9154 / 0.8711 | [model](TODO) &#124; [log](TODO) |
| edsr_x4c64b16_g1_300k_div2k | 30.2223 / 0.8500 | 26.7870 / 0.7366 | 28.9675 / 0.8172 | [model](TODO) &#124; [log](TODO) |
