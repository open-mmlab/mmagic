# BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

## Introduction

```
@article{chan2021basicvsr,
          author = {Chan, Kelvin CK and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
          title = {BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond},
          journal = {Proceedings of the IEEE conference on computer vision and pattern recognition},
          year = {2021}
}
```

## Results and Models

Evaluated on RGB channels for REDS4 and Y channel for others.
The metrics are `PSNR`/`SSIM`.

| Method               | REDS4 (BIx4) | Vimeo-90K-T (BIx4) |   Vid4 (BIx4)  |  UDM10 (BDx4)  | Vimeo-90K-T (BDx4) |   Vid4 (BDx4)  |   Download   |
|----------------------|:------------:|:------------------:|:--------------:|:--------------:|:------------------:|:--------------:|:------------:|
| basicvsr_reds4       |              |          -         |        -       |        -       |          -         |        -       | model \| log |
| basicvsr_vimeo90k_bi |       -      |   37.2299/0.9447   | 27.2296/0.8227 |        -       |          -         |        -       | model \| log |
| basicvsr_vimeo90k_bd |       -      |          -         |        -       | 39.8802/0.9683 |   37.5730/0.9495   | 27.9278/0.8537 | model \| log |
| spynet               |       -      |          -         |        -       |        -       |          -         |        -       |     model    |
