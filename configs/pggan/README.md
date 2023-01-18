# PGGAN (ICLR'2018)

> [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)

> **Task**: Unconditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024^2. We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CelebA dataset.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143053374-c03894c3-6def-49c2-94ed-80c4accee726.JPG" />
</div>

## Results and models

<div align="center">
  <b> Results (compressed) from our PGGAN trained in CelebA-HQ@1024</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/114009864-1df45400-9896-11eb-9d25-da9eabfe02ce.png" width="800"/>
</div>

|                              Model                              |    Dataset     | MS-SSIM |     SWD(xx,xx,xx,xx/avg)     |                                         Download                                         |
| :-------------------------------------------------------------: | :------------: | :-----: | :--------------------------: | :--------------------------------------------------------------------------------------: |
| [pggan_128x128](./pggan_8xb4-12Mimgs_celeba-cropped-128x128.py) | celeba-cropped | 0.3023  | 3.42, 4.04, 4.78, 20.38/8.15 | [model](https://download.openmmlab.com/mmediting/pggan/pggan_celeba-cropped_128_g8_20210408_181931-85a2e72c.pth) |
|  [pggan_128x128](./pggan_8xb4-12Mimgs_lsun-bedroom-128x128.py)  |  lsun-bedroom  | 0.0602  |  3.5, 2.96, 2.76, 9.65/4.72  | [model](https://download.openmmlab.com/mmediting/pggan/pggan_lsun-bedroom_128x128_g8_20210408_182033-5e59f45d.pth) |
|  [pggan_1024x1024](./pggan_8xb4-12Mimg_celeba-hq-1024x1024.py)  |   celeba-hq    | 0.3379  | 8.93, 3.98, 3.07, 2.64/4.655 | [model](https://download.openmmlab.com/mmediting/pggan/pggan_celeba-hq_1024_g8_20210408_181911-f1ef51c3.pth) |

## Citation

<summary align="right"><a href="https://arxiv.org/abs/1710.10196">PGGAN (arXiv'2017)</a></summary>

```latex
@article{karras2017progressive,
  title={Progressive growing of gans for improved quality, stability, and variation},
  author={Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  journal={arXiv preprint arXiv:1710.10196},
  year={2017},
  url={https://arxiv.org/abs/1710.10196},
}
```
