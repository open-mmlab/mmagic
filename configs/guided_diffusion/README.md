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

| Model | Dataset          | Sampling Scheduler | Steps     | Guidance Scale | FID  | Config | Ckpt |
| ----- | ---------------- | ------------------ | --------- | -------------- | ---- | ------ | ---- |
| ADM   | ImageNet 64x64   | DDPM               | 250 steps | -              | 2.61 |        |      |
| ADM-G | ImageNet 128x128 |                    |           |                | 2.97 |        |      |
| ADM-G | ImageNet 256x256 | DDPM               | 250 steps | 1.0            | 4.59 |        |      |
| ADM-G | ImageNet 256x256 | DDIM               | 25 steps  | 1.0            | 5.44 |        |      |
| ADM-G | ImageNet 512x512 |                    |           |                | 7.72 |        |      |

## Quick Start

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/guided_diffusion/adm-u_ddim250_8xb32_imagenet-64x64.py https://download.openmmlab.com/mmgen/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth

# single-gpu test
python tools/test.py configs/guided_diffusion/adm-u_ddim250_8xb32_imagenet-64x64.py https://download.openmmlab.com/mmgen/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth

# multi-gpu test
./tools/dist_test.sh configs/guided_diffusion/adm-u_ddim250_8xb32_imagenet-64x64.py https://download.openmmlab.com/mmgen/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>

## Citation

```bibtex
@article{PrafullaDhariwal2021DiffusionMB,
  title={Diffusion Models Beat GANs on Image Synthesis},
  author={Prafulla Dhariwal and Alex Nichol},
  journal={arXiv: Learning},
  year={2021}
}
```
