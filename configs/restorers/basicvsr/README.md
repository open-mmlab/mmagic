# BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

## Introduction

```
@InProceedings{chan2021basicvsr,
  author = {Chan, Kelvin CK and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title = {BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

## Results and Models

Evaluated on RGB channels for REDS4 and Y channel for others. The metrics are `PSNR`/`SSIM`.

The pretrained weights of SPyNet can be found [here](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-322d39be.pth)   .

| Method                                                                                                                         |  REDS4 (BIx4)  | Vimeo-90K-T (BIx4) |   Vid4 (BIx4)  |                                                                                                              Download                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------|:--------------:|:------------------:|:--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [basicvsr_reds4](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds.py)              | 31.4170/0.8909 |          -         |        -       |       [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-b4b03f4d.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20210409_092646.log.json)       |
| [basicvsr_vimeo90k_bi](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_vimeo90k_bi.py) |        -       |   37.2299/0.9447   | 27.2296/0.8227 | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409-ef89bf61.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409_132702.log.json) |

<br />

| Method                                                                                                                         |  UDM10 (BDx4)  | Vimeo-90K-T (BDx4) |   Vid4 (BDx4)  |                                                                                                               Download                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------|:--------------:|:------------------:|:--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [basicvsr_vimeo90k_bd](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_vimeo90k_bd.py) |        39.8802/0.9683 |   37.5730/0.9495   | 27.9278/0.8537 | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409-b5a982fc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409_132740.log.json) |
