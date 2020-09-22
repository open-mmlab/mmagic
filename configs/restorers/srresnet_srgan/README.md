# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

## Introduction

```
@inproceedings{ledig2016photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  year={2016}
}
```

## Results and Models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.

The metrics are `PSNR / SSIM`.

|               Method               |       Set5        |      Set14       |      DIV2K       |                                                                                                                                  Download                                                                                                                                   |
| :--------------------------------: | :---------------: | :--------------: | :--------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| msrresnet_x4c64b16_1x16_300k_div2k | 30.2252 / 0.8491  | 26.7762 / 0.7369 | 28.9748 / 0.8178 | [model](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521-61556be5.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521_110246.log.json) |
|  srgan_x4c64b16_1x16_1000k_div2k   | 27.9499 /  0.7846 | 24.7383 / 0.6491 | 26.5697 / 0.7365 |    [model](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200606-a1f0810e.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200506_191442.log.json)    |
