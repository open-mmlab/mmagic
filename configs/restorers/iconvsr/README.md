# BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond

## Introduction

<!-- [ALGORITHM] -->

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

The pretrained weights of the IconVSR components can be found here: [SPyNet](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth), [EDVR-M for REDS](https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_reds_20210413-3867262f.pth), and [EDVR-M for Vimeo-90K](https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_vimeo90k_20210413-e40e99a8.pth)    .


|                                                            Method                                                           | REDS4 (BIx4)<br>PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4)<br>PSNR/SSIM (Y) | Vid4 (BIx4)<br>PSNR/SSIM (Y) | UDM10 (BDx4)<br>PSNR/SSIM (Y) | Vimeo-90K-T (BDx4)<br>PSNR/SSIM (Y) | Vid4 (BDx4)<br>PSNR/SSIM (Y) |                                                                                                             Download                                                                                                            |
|:---------------------------------------------------------------------------------------------------------------------------:|:-------------------------------:|:-----------------------------------:|:----------------------------:|:-----------------------------:|:-----------------------------------:|:----------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        [iconvsr_reds4](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/iconvsr/iconvsr_reds.py)       |        **31.6926/0.8951**       |            36.4963/0.9410           |      **27.4350/0.8334**      |         35.3336/0.9463        |            34.4626/0.9280           |        25.2045/0.7721        |       [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-7b1eb65b.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413_222735.log.json)       |
| [iconvsr_vimeo90k_bi](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/iconvsr/iconvsr_vimeo90k_bi.py) |          30.3452/0.8659         |          **37.4008/0.9464**         |        27.3738/0.8274        |         34.2639/0.9388        |            34.6193/0.9293           |        24.6648/0.7480        | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413-88ab5619.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413_222757.log.json) |
| [iconvsr_vimeo90k_bd](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/iconvsr/iconvsr_vimeo90k_bd.py) |          29.0150/0.8465         |            34.7454/0.9336           |        26.2763/0.8006        |       **39.9403/0.8006**      |          **37.7794/0.9513**         |      **28.1916/0.8592**      | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414-04d713dc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414_084128.log.json) |
