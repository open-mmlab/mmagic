# AOT-GAN (TVCG'2021)

> [AOT-GAN: Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/pdf/2104.01431.pdf)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

State-of-the-art image inpainting approaches can suffer from generating distorted structures and blurry textures in high-resolution images (e.g., 512x512). The challenges mainly drive from (1) image content reasoning from distant contexts, and (2) fine-grained texture synthesis for a large missing region. To overcome these two challenges, we propose an enhanced GAN-based model, named Aggregated COntextual-Transformation GAN (AOT-GAN), for high-resolution image inpainting. Specifically, to enhance context reasoning, we construct the generator of AOT-GAN by stacking multiple layers of a proposed AOT block. The AOT blocks aggregate contextual transformations from various receptive fields, allowing to capture both informative distant image contexts and rich patterns of interest for context reasoning. For improving texture synthesis, we enhance the discriminator of AOT-GAN by training it with a tailored mask-prediction task. Such a training objective forces the discriminator to distinguish the detailed appearances of real and synthesized patches, and in turn, facilitates the generator to synthesize clear textures. Extensive comparisons on Places2, the most challenging benchmark with 1.8 million high-resolution images of 365 complex scenes, show that our model outperforms the state-of-the-art by a significant margin in terms of FID with 38.60% relative improvement. A user study including more than 30 subjects further validates the superiority of AOT-GAN. We further evaluate the proposed AOT-GAN in practical applications, e.g., logo removal, face editing, and object removal. Results show that our model achieves promising completions in the real world. We release code and models in [this https URL](https://github.com/researchmm/AOT-GAN-for-Inpainting).

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12756472/169230414-3ca7fb6b-cf2a-401f-8696-71df75a08c32.png"/>
</div >

## Results and models

**Places365-Challenge**

|                              Method                              |     Mask Type      | Resolution | Train Iters |   Test Set    | l1 error | PSNR  | SSIM  |                              Download                              |
| :--------------------------------------------------------------: | :----------------: | :--------: | :---------: | :-----------: | :------: | :---: | :---: | :----------------------------------------------------------------: |
| [AOT-GAN](/configs/inpainting/AOT-GAN/AOT-GAN_512x512_4x12_places.py) | free-form (50-60%) |  512x512   |    500k     | Places365-val |   7.07   | 19.01 | 0.682 | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmediting/inpainting/aot_gan/AOT-GAN_512x512_4x12_places_20220509-6641441b.pth) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmediting/inpainting/aot_gan/AOT-GAN_512x512_4x12_places_20220509-6641441b.json) |

More results for different mask area:

<!-- SKIP THIS TABLE -->

| Metric          | Mask Area | Paper Results | Reimplemented Results |
| :-------------- | :-------- | :------------ | :-------------------- |
| L1 (10^-2)      | 1 – 10%   | 0.55          | 0.54                  |
| (lower better)  | 10 – 20%  | 1.19          | 1.47                  |
|                 | 20 – 30%  | 2.11          | 2.79                  |
|                 | 30 – 40%  | 3.20          | 4.38                  |
|                 | 40 – 50%  | 4.51          | 6.28                  |
|                 | 50 – 60%  | 7.07          | 10.16                 |
| PSNR            | 1 – 10%   | 34.79         | inf                   |
| (higher better) | 10 – 20%  | 29.49         | 31.22                 |
|                 | 20 – 30%  | 26.03         | 27.65                 |
|                 | 30 – 40%  | 23.58         | 25.06                 |
|                 | 40 – 50%  | 21.65         | 23.01                 |
|                 | 50 – 60%  | 19.01         | 20.05                 |
| SSIM            | 1 – 10%   | 0.976         | 0.982                 |
| (higher better) | 10 – 20%  | 0.940         | 0.951                 |
|                 | 20 – 30%  | 0.890         | 0.911                 |
|                 | 30 – 40%  | 0.835         | 0.866                 |
|                 | 40 – 50%  | 0.773         | 0.815                 |
|                 | 50 – 60%  | 0.682         | 0.739                 |

## Citation

```bibtex
@inproceedings{yan2021agg,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Aggregated Contextual Transformations for High-Resolution Image Inpainting},
  booktitle = {Arxiv},
  pages={-},
  year = {2020}
}
```
