# GLEAN (CVPR'2021)

> [GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution](https://arxiv.org/abs/2012.00739)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We show that pre-trained Generative Adversarial Networks (GANs), e.g., StyleGAN, can be used as a latent bank to improve the restoration quality of large-factor image super-resolution (SR). While most existing SR approaches attempt to generate realistic textures through learning with adversarial loss, our method, Generative LatEnt bANk (GLEAN), goes beyond existing practices by directly leveraging rich and diverse priors encapsulated in a pre-trained GAN. But unlike prevalent GAN inversion methods that require expensive image-specific optimization at runtime, our approach only needs a single forward pass to generate the upscaled image. GLEAN can be easily incorporated in a simple encoder-bank-decoder architecture with multi-resolution skip connections. Switching the bank allows the method to deal with images from diverse categories, e.g., cat, building, human face, and car. Images upscaled by GLEAN show clear improvements in terms of fidelity and texture faithfulness in comparison to existing methods.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144019196-2642f3be-f82e-4fa4-8d96-4161354db9a7.png" width="400"/>
</div >

## Results and models

For the meta info used in training and test, please refer to [here](https://github.com/ckkelvinchan/GLEAN). The results are evaluated on RGB channels.

|                                              Method                                              | PSNR  |                                              Download                                               |
| :----------------------------------------------------------------------------------------------: | :---: | :-------------------------------------------------------------------------------------------------: |
|                     [glean_cat_8x](/configs/restorers/glean/glean_cat_8x.py)                     | 23.98 | [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614-d3ac8683.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614_145540.log.json) |
|                   [glean_ffhq_16x](/configs/restorers/glean/glean_ffhq_16x.py)                   | 26.91 | [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527-61a3afad.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527_194536.log.json) |
|                    [glean_cat_16x](/configs/restorers/glean/glean_cat_16x.py)                    | 20.88 | [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527-68912543.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527_103708.log.json) |
| [glean_in128out1024_4x2_300k_ffhq_celebahq](/configs/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq.py) | 27.94 | [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812_100549.log.json) |

## Citation

```bibtex
@InProceedings{chan2021glean,
  author = {Chan, Kelvin CK and Wang, Xintao and Xu, Xiangyu and Gu, Jinwei and Loy, Chen Change},
  title = {GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```
