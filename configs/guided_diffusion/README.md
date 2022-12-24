# Guided Diffusion (NeurIPS'2021)

> [Diffusion Models Beat GANs on Image Synthesis](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)

> **Task**: Image Generation

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models. We achieve this on unconditional image synthesis by finding a better architecture through a series of ablations. For conditional image synthesis, we further improve sample quality with classifier guidance: a simple, compute-efficient method for trading off diversity for fidelity using gradients from a classifier. We achieve an FID of 2.97 on ImageNet 128x128, 4.59 on ImageNet 256x256, and 7.72 on ImageNet 512x512, and we match BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution. Finally, we find that classifier guidance combines well with upsampling diffusion models, further improving FID to 3.94 on ImageNet 256x256 and 3.85 on ImageNet 512x512.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/22982797/204706276-e340c545-3ec6-48bf-be21-58ed44e8a4df.jpg" width="400"/>
</div >

## Results and models

**ImageNet**

| Method | Resolution | Config                                      | Weights                                                                                                            |
| ------ | ---------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| adm-u  | 64x64      | [config](./adm-u_8xb32_imagenet-64x64.py)   | [model](https://download.openmmlab.com/mmediting/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth) |
| adm-u  | 512x512    | [config](./adm-u_8xb32_imagenet-512x512.py) | [model](https://download.openmmlab.com/mmediting/guided_diffusion/adm-u_8xb32_imagenet-512x512-60b381cb.pth)       |

**Note** To support disco diffusion, we support guided diffusion briefly. Complete support of guided diffusion with metrics and test/train logs will come soom!

## Quick Start

Coming soon!

## Citation

```bibtex
@article{PrafullaDhariwal2021DiffusionMB,
  title={Diffusion Models Beat GANs on Image Synthesis},
  author={Prafulla Dhariwal and Alex Nichol},
  journal={arXiv: Learning},
  year={2021}
}
```
