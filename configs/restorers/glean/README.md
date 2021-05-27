# GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@InProceedings{chan2021glean,
  author = {Chan, Kelvin CK and Wang, Xintao and Xu, Xiangyu and Gu, Jinwei and Loy, Chen Change},
  title = {GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

## Meta info
For the meta info used in training and test, please refer to [here](https://github.com/ckkelvinchan/GLEAN).

## Results
The results are evaluated on RGB channels.


|                                                      Method                                                     |  PSNR |                                                                                                         Download                                                                                                         |
|:---------------------------------------------------------------------------------------------------------------:|:-----:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [glean_ffhq_16x](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/glean/glean_ffhq_16x.py) | 26.91 |     [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527-61a3afad.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527_194536.log.json)    |
|                                                [glean_cat_16x](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/glean/glean_cat_16x.py)                                                | 20.88 | [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527-68912543.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527_103708.log.json) |
