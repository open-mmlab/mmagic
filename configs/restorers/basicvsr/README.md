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


|                                                             Method                                                             | REDS4 (BIx4)<br>PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4)<br>PSNR/SSIM (Y) | Vid4 (BIx4)<br>PSNR/SSIM (Y) | UDM10 (BDx4)<br>PSNR/SSIM (Y) | Vimeo-90K-T (BDx4)<br>PSNR/SSIM (Y) | Vid4 (BDx4)<br>PSNR/SSIM (Y) |                                                                                                               Download                                                                                                              |
|:------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------:|:-----------------------------------:|:----------------------------:|:-----------------------------:|:-----------------------------------:|:----------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        [basicvsr_reds4](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds.py)       |        **31.4170/0.8909**       |            36.2870/0.9388           |        27.2223/0.8298        |         33.4510/0.9297        |            34.5053/0.9280           |        24.4390/0.7441        |       [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-b4b03f4d.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20210409_092646.log.json)       |
| [basicvsr_vimeo90k_bi](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_vimeo90k_bi.py) |          30.3128/0.8660         |          **37.2299/0.9447**         |      **27.2296/0.8227**      |         34.5488/0.9423        |            34.8713/0.9313           |        25.0377/0.7622        | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409-ef89bf61.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409_132702.log.json) |
| [basicvsr_vimeo90k_bd](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_vimeo90k_bd.py) |          29.0376/0.8481         |            34.7094/0.9332           |        26.2356/0.8000        |       **39.8802/0.9683**      |          **37.5730/0.9495**         |      **27.9278/0.8537**      | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409-b5a982fc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409_132740.log.json) |
