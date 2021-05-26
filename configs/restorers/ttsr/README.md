# Learning Texture Transformer Network for Image Super-Resolution

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{yang2020learning,
  title={Learning texture transformer network for image super-resolution},
  author={Yang, Fuzhi and Yang, Huan and Fu, Jianlong and Lu, Hongtao and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5791--5800},
  year={2020}
}
```

## Results

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.

The metrics are `PSNR / SSIM`.

|                                         Method                                                  | scale |       CUFED      |                                                                                                                   Download                                                                                                                                                                                                                                                                                                                  |
| :---------------------------------------------------------------------------------------------: | :---: | :--------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [ttsr-rec_x4_c64b16_g1_200k_CUFED](/configs/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED.py) |   x4  | 25.2433 / 0.7491 | [model](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.pth?versionId=CAEQKxiBgIDht5ONzRciIDdjZTQ1NmFmYzhjNjQ5NGFhNjkyNzU1N2UxMjMyZWE4) \| [log](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.log.json?versionId=CAEQKxiCgMCnuJONzRciIDUzNmVkNGNmNTlkMDQzMmFhZDAzYzQ5NmUzNTI5YmYz) |
