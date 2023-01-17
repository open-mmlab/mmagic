# GGAN (ArXiv'2017)

> [Geometric GAN](https://arxiv.org/abs/1705.02894)

> **Task**: Unconditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Generative Adversarial Nets (GANs) represent an important milestone for effective generative models, which has inspired numerous variants seemingly different from each other. One of the main contributions of this paper is to reveal a unified geometric structure in GAN and its variants. Specifically, we show that the adversarial generative model training can be decomposed into three geometric steps: separating hyperplane search, discriminator parameter update away from the separating hyperplane, and the generator update along the normal vector direction of the separating hyperplane. This geometric intuition reveals the limitations of the existing approaches and leads us to propose a new formulation called geometric GAN using SVM separating hyperplane that maximizes the margin. Our theoretical analysis shows that the geometric GAN converges to a Nash equilibrium between the discriminator and generator. In addition, extensive numerical results show that the superior performance of geometric GAN.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143051600-6a3e5c37-259e-4b77-a847-c6ad1eafa65f.JPG"/>
</div>

## Results and models

<div align="center">
  <b> GGAN 64x64, CelebA-Cropped</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/116691577-9067d800-a9ed-11eb-8ea4-be79884d8502.PNG" width="800"/>
</div>

|                                 Model                                 |    Dataset     |               SWD               | MS-SSIM |   FID   |                                 Download                                 |
| :-------------------------------------------------------------------: | :------------: | :-----------------------------: | :-----: | :-----: | :----------------------------------------------------------------------: |
| [GGAN 64x64](./ggan_dcgan-archi_lr1e-3-1xb128-12Mimgs_celeba-cropped-64x64.py) | CelebA-Cropped |    11.18, 12.21, 39.16/20.85    | 0.3318  | 20.1797 | [model](https://download.openmmlab.com/mmediting/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-3_64_b128x1_12m.pth)  \| [log](https://download.openmmlab.com/mmediting/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-3_64_b128x1_12m_20210430_113839.log.json) |
| [GGAN 128x128](./ggan_dcgan-archi_lr1e-4-1xb64-10Mimgs_celeba-cropped-128x128.py) | CelebA-Cropped | 9.81, 11.29, 19.22, 47.79/22.03 | 0.3149  | 18.7647 | [model](https://download.openmmlab.com/mmediting/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-4_128_b64x1_10m_20210430_143027-516423dc.pth) \| [log](https://download.openmmlab.com/mmediting/ggan/ggan_celeba-cropped_dcgan-archi_lr-1e-4_128_b64x1_10m_20210423_154258.log.json) |
| [GGAN 64x64](./ggan_lsgan-archi_lr1e-4-1xb128-20Mimgs_lsun-bedroom-64x64.py) |  LSUN-Bedroom  |      9.1, 6.2, 12.27/9.19       | 0.0649  | 39.9261 | [model](https://download.openmmlab.com/mmediting/ggan/ggan_lsun-bedroom_lsgan_archi_lr-1e-4_64_b128x1_20m_20210430_143114-5d99b76c.pth) \| [log](https://download.openmmlab.com/mmediting/ggan/ggan_lsun-bedroom_lsgan_archi_lr-1e-4_64_b128x1_20m_20210428_202027.log.json) |

Note: In the original implementation of [GGAN](https://github.com/lim0606/pytorch-geometric-gan), they set `G_iters` to 10. However our framework does not support `G_iters` currently, so we dropped the settings in the original implementation and conducted several experiments with our own settings. We have shown above the experiment results with the lowest `fid` score. \
Original settings and our settings:

<!-- SKIP THIS TABLE -->

|       Model        |    Dataset     | Architecture | optimizer |  lr_G  |  lr_D  | G_iters | D_iters |
| :----------------: | :------------: | :----------: | :-------: | :----: | :----: | :-----: | :-----: |
| GGAN(origin) 64x64 | CelebA-Cropped | dcgan-archi  |  RMSprop  | 0.0002 | 0.0002 |   10    |    1    |
|  GGAN(ours) 64x64  | CelebA-Cropped | dcgan-archi  |   Adam    | 0.001  | 0.001  |    1    |    1    |
| GGAN(origin) 64x64 |  LSUN-Bedroom  | dcgan-archi  |  RMSprop  | 0.0002 | 0.0002 |   10    |    1    |
|  GGAN(ours) 64x64  |  LSUN-Bedroom  | lsgan-archi  |   Adam    | 0.0001 | 0.0001 |    1    |    1    |

## Citation

```latex
@article{lim2017geometric,
  title={Geometric gan},
  author={Lim, Jae Hyun and Ye, Jong Chul},
  journal={arXiv preprint arXiv:1705.02894},
  year={2017},
  url={https://arxiv.org/abs/1705.02894},
}
```
