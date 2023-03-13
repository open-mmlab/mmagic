# Positional Encoding in GANs (CVPR'2021)

> [Positional Encoding as Spatial Inductive Bias in GANs](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_Positional_Encoding_As_Spatial_Inductive_Bias_in_GANs_CVPR_2021_paper.html)

> **Task**: Unconditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

SinGAN shows impressive capability in learning internal patch distribution despite its limited effective receptive field. We are interested in knowing how such a translation-invariant convolutional generator could capture the global structure with just a spatially i.i.d. input. In this work, taking SinGAN and StyleGAN2 as examples, we show that such capability, to a large extent, is brought by the implicit positional encoding when using zero padding in the generators. Such positional encoding is indispensable for generating images with high fidelity. The same phenomenon is observed in other generative architectures such as DCGAN and PGGAN. We further show that zero padding leads to an unbalanced spatial bias with a vague relation between locations. To offer a better spatial inductive bias, we investigate alternative positional encodings and analyze their effects. Based on a more flexible positional encoding explicitly, we propose a new multi-scale training strategy and demonstrate its effectiveness in the state-of-the-art unconditional generator StyleGAN2. Besides, the explicit spatial inductive bias substantially improve SinGAN for more versatile image manipulation.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143053767-c6a503b2-87ff-434a-a439-d9fb0e98d804.JPG"/>
</div>

## Results and models for MS-PIE

<div align="center">
  <b> 896x896 results generated from a 256 generator using MS-PIE</b>
  <br/>
  <img src="https://download.openmmlab.com/mmediting/pe_in_gans/mspie_256-896_demo.png" width="800"/>
</div>

|                                 Model                                  | Dataset | Reference in Paper |     Scales     | FID50k |   P&R10k    |                                  Download                                  |
| :--------------------------------------------------------------------: | :-----: | :----------------: | :------------: | :----: | :---------: | :------------------------------------------------------------------------: |
| [stylegan2_c2_8xb3-1100kiters_ffhq-256x256](./stylegan2_c2_8xb3-1100kiters_ffhq-256x256.py) |  FFHQ   |   Tab.5 config-a   |      256       |  5.56  | 75.92/51.24 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/stylegan2_c2_config-a_ffhq_256x256_b3x8_1100k_20210406_145127-71d9634b.pth) |
| [stylegan2_c2_8xb3-1100kiters_ffhq-512x512](./stylegan2_c2_8xb3-1100kiters_ffhq-512x512.py) |  FFHQ   |   Tab.5 config-b   |      512       |  4.91  | 75.65/54.58 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/stylegan2_c2_config-b_ffhq_512x512_b3x8_1100k_20210406_145142-e85e5cf4.pth) |
| [mspie-stylegan2-config-c_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-c_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-c   | 256, 384, 512  |  3.35  | 73.84/55.77 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-c_ffhq_256-512_b3x8_1100k_20210406_144824-9f43b07d.pth) |
| [mspie-stylegan2-config-d_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-d_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-d   | 256, 384, 512  |  3.50  | 73.28/56.16 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-d_ffhq_256-512_b3x8_1100k_20210406_144840-dbefacf6.pth) |
| [mspie-stylegan2-config-e_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-e_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-e   | 256, 384, 512  |  3.15  | 74.13/56.88 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-e_ffhq_256-512_b3x8_1100k_20210406_144906-98d5a42a.pth) |
| [mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-f   | 256, 384, 512  |  2.93  | 73.51/57.32 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-512_b3x8_1100k_20210406_144927-4f4d5391.pth) |
| [mspie-stylegan2-config-g_c1_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-g_c1_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-g   | 256, 384, 512  |  3.40  | 73.05/56.45 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c1_config-g_ffhq_256-512_b3x8_1100k_20210406_144758-2df61752.pth) |
| [mspie-stylegan2-config-h_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-h_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-h   | 256, 384, 512  |  4.01  | 72.81/54.35 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-h_ffhq_256-512_b3x8_1100k_20210406_145006-84cf3f48.pth) |
| [mspie-stylegan2-config-i_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-i_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   | 2  Tab.5 config-i  |  56, 384, 512  |  3.76  | 73.26/54.71 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-i_ffhq_256-512_b3x8_1100k_20210406_145023-c2b0accf.pth) |
| [mspie-stylegan2-config-j_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-j_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-j   | 256, 384, 512  |  4.23  | 73.11/54.63 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-j_ffhq_256-512_b3x8_1100k_20210406_145044-c407481b.pth) |
| [mspie-stylegan2-config-k_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-k_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-k   | 256, 384, 512  |  4.17  | 73.05/51.07 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-k_ffhq_256-512_b3x8_1100k_20210406_145105-6d8cc39f.pth) |
| [mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-896](./mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-896.py) |  FFHQ   | higher-resolution  | 256, 512, 896  |  4.10  | 72.21/50.29 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-896_b3x8_1100k_20210406_144943-6c18ad5d.pth) |
| [mspie-stylegan2-config-f_c1_8xb2-1600kiters_ffhq-256-1024](./mspie-stylegan2-config-f_c1_8xb2-1600kiters_ffhq-256-1024.py) |  FFHQ   | higher-resolution  | 256, 512, 1024 |  6.24  | 71.79/49.92 | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c1_config-f_ffhq_256-1024_b2x8_1600k_20210406_144716-81cbdc96.pth) |

|                               Model                               | Dataset | Reference in Paper |     Scales     | FID50k | Precision10k | Recall10k |                               Download                                |
| :---------------------------------------------------------------: | :-----: | :----------------: | :------------: | :----: | :----------: | :-------: | :-------------------------------------------------------------------: |
| [stylegan2_c2_8xb3-1100kiters_ffhq-256x256](./stylegan2_c2_8xb3-1100kiters_ffhq-256x256.py) |  FFHQ   |   Tab.5 config-a   |      256       |  5.56  |    75.92     |   51.24   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/stylegan2_c2_config-a_ffhq_256x256_b3x8_1100k_20210406_145127-71d9634b.pth) |
| [stylegan2_c2_8xb3-1100kiters_ffhq-512x512](./stylegan2_c2_8xb3-1100kiters_ffhq-512x512.py) |  FFHQ   |   Tab.5 config-b   |      512       |  4.91  |    75.65     |   54.58   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/stylegan2_c2_config-b_ffhq_512x512_b3x8_1100k_20210406_145142-e85e5cf4.pth) |
| [mspie-stylegan2-config-c_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-c_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-c   | 256, 384, 512  |  3.35  |    73.84     |   55.77   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-c_ffhq_256-512_b3x8_1100k_20210406_144824-9f43b07d.pth) |
| [mspie-stylegan2-config-d_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-d_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-d   | 256, 384, 512  |  3.50  |    73.28     |   56.16   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-d_ffhq_256-512_b3x8_1100k_20210406_144840-dbefacf6.pth) |
| [mspie-stylegan2-config-e_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-e_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-e   | 256, 384, 512  |  3.15  |    74.13     |   56.88   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-e_ffhq_256-512_b3x8_1100k_20210406_144906-98d5a42a.pth) |
| [mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-f   | 256, 384, 512  |  2.93  |    73.51     |   57.32   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-512_b3x8_1100k_20210406_144927-4f4d5391.pth) |
| [mspie-stylegan2-config-g_c1_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-g_c1_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-g   | 256, 384, 512  |  3.40  |    73.05     |   56.45   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c1_config-g_ffhq_256-512_b3x8_1100k_20210406_144758-2df61752.pth) |
| [mspie-stylegan2-config-h_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-h_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-h   | 256, 384, 512  |  4.01  |    72.81     |   54.35   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-h_ffhq_256-512_b3x8_1100k_20210406_145006-84cf3f48.pth) |
| [mspie-stylegan2-config-i_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-i_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-i   | 256, 384, 512  |  3.76  |    73.26     |   54.71   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-i_ffhq_256-512_b3x8_1100k_20210406_145023-c2b0accf.pth) |
| [mspie-stylegan2-config-j_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-j_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-j   | 256, 384, 512  |  4.23  |    73.11     |   54.63   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-j_ffhq_256-512_b3x8_1100k_20210406_145044-c407481b.pth) |
| [mspie-stylegan2-config-k_c2_8xb3-1100kiters_ffhq-256-512](./mspie-stylegan2-config-k_c2_8xb3-1100kiters_ffhq-256-512.py) |  FFHQ   |   Tab.5 config-k   | 256, 384, 512  |  4.17  |    73.05     |   51.07   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-k_ffhq_256-512_b3x8_1100k_20210406_145105-6d8cc39f.pth) |
| [mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-896](./mspie-stylegan2-config-f_c2_8xb3-1100kiters_ffhq-256-896.py) |  FFHQ   | higher-resolution  | 256, 512, 896  |  4.10  |    72.21     |   50.29   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c2_config-f_ffhq_256-896_b3x8_1100k_20210406_144943-6c18ad5d.pth) |
| [mspie-stylegan2-config-f_c1_8xb2-1600kiters_ffhq-256-1024](./mspie-stylegan2-config-f_c1_8xb2-1600kiters_ffhq-256-1024.py) |  FFHQ   | higher-resolution  | 256, 512, 1024 |  6.24  |    71.79     |   49.92   | [model](https://download.openmmlab.com/mmediting/pe_in_gans/mspie-stylegan2_c1_config-f_ffhq_256-1024_b2x8_1600k_20210406_144716-81cbdc96.pth) |

Note that we report the FID and P&R metric (FFHQ dataset) in the largest scale.

## Results and Models for SinGAN

<div align="center">
  <b> Positional Encoding in SinGAN</b>
  <br/>
  <img src="https://nbei.github.io/gan-pos-encoding/teaser-web-singan.png" width="800"/>
</div>

|                             Model                             |                             Dataset                              | Num Scales |                             Download                              |
| :-----------------------------------------------------------: | :--------------------------------------------------------------: | :--------: | :---------------------------------------------------------------: |
| [singan_interp-pad_balloons](./singan_interp-pad_balloons.py) | [balloons.png](https://download.openmmlab.com/mmediting/dataset/singan/balloons.png) |     8      | [ckpt](https://download.openmmlab.com/mmediting/pe_in_gans/singan_interp-pad_balloons_20210406_180014-96f51555.pth) \| [pkl](https://download.openmmlab.com/mmediting/pe_in_gans/singan_interp-pad_balloons_20210406_180014-96f51555.pkl) |
| [singan_interp-pad_disc-nobn_balloons](./singan_interp-pad_disc-nobn_balloons.py) | [balloons.png](https://download.openmmlab.com/mmediting/dataset/singan/balloons.png) |     8      | [ckpt](https://download.openmmlab.com/mmediting/pe_in_gans/singan_interp-pad_disc-nobn_balloons_20210406_180059-7d63e65d.pth) \| [pkl](https://download.openmmlab.com/mmediting/pe_in_gans/singan_interp-pad_disc-nobn_balloons_20210406_180059-7d63e65d.pkl) |
| [singan_interp-pad_disc-nobn_fish](./singan_interp-pad_disc-nobn_fish.py) | [fish.jpg](https://download.openmmlab.com/mmediting/dataset/singan/fish-crop.jpg) |     10     | [ckpt](https://download.openmmlab.com/mmediting/pe_in_gans/singan_interp-pad_disc-nobn_fis_20210406_175720-9428517a.pth) \| [pkl](https://download.openmmlab.com/mmediting/pe_in_gans/singan_interp-pad_disc-nobn_fis_20210406_175720-9428517a.pkl) |
|            [singan-csg_fish](./singan-csg_fish.py)            | [fish.jpg](https://download.openmmlab.com/mmediting/dataset/singan/fish-crop.jpg) |     10     | [ckpt](https://download.openmmlab.com/mmediting/pe_in_gans/singan_csg_fis_20210406_175532-f0ec7b61.pth) \| [pkl](https://download.openmmlab.com/mmediting/pe_in_gans/singan_csg_fis_20210406_175532-f0ec7b61.pkl) |
|        [singan-csg_bohemian](./singan-csg_bohemian.py)        | [bohemian.png](https://download.openmmlab.com/mmediting/dataset/singan/bohemian.png) |     10     | [ckpt](https://download.openmmlab.com/mmediting/pe_in_gans/singan_csg_bohemian_20210407_195455-5ed56db2.pth) \| [pkl](https://download.openmmlab.com/mmediting/pe_in_gans/singan_csg_bohemian_20210407_195455-5ed56db2.pkl) |
|       [singan_spe-dim4_fish](./singan_spe-dim4_fish.py)       | [fish.jpg](https://download.openmmlab.com/mmediting/dataset/singan/fish-crop.jpg) |     10     | [ckpt](https://download.openmmlab.com/mmediting/pe_in_gans/singan_spe-dim4_fish_20210406_175933-f483a7e3.pth) \| [pkl](https://download.openmmlab.com/mmediting/pe_in_gans/singan_spe-dim4_fish_20210406_175933-f483a7e3.pkl) |
|   [singan_spe-dim4_bohemian](./singan_spe-dim4_bohemian.py)   | [bohemian.png](https://download.openmmlab.com/mmediting/dataset/singan/bohemian.png) |     10     | [ckpt](https://download.openmmlab.com/mmediting/pe_in_gans/singan_spe-dim4_bohemian_20210406_175820-6e484a35.pth) \| [pkl](https://download.openmmlab.com/mmediting/pe_in_gans/singan_spe-dim4_bohemian_20210406_175820-6e484a35.pkl) |
|   [singan_spe-dim8_bohemian](./singan_spe-dim8_bohemian.py)   | [bohemian.png](https://download.openmmlab.com/mmediting/dataset/singan/bohemian.png) |     10     | [ckpt](https://download.openmmlab.com/mmediting/pe_in_gans/singan_spe-dim8_bohemian_20210406_175858-7faa50f3.pth) \| [pkl](https://download.openmmlab.com/mmediting/pe_in_gans/singan_spe-dim8_bohemian_20210406_175858-7faa50f3.pkl) |

## Citation

```latex
@article{xu2020positional,
  title={Positional Encoding as Spatial Inductive Bias in GANs},
  author={Xu, Rui and Wang, Xintao and Chen, Kai and Zhou, Bolei and Loy, Chen Change},
  journal={arXiv preprint arXiv:2012.05217},
  year={2020},
  url={https://openaccess.thecvf.com/content/CVPR2021/html/Xu_Positional_Encoding_As_Spatial_Inductive_Bias_in_GANs_CVPR_2021_paper.html},
}
```
