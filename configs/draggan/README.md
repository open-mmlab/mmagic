# DragGAN (SIGGRAPH'2023)

> [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://arxiv.org/pdf/2305.10973.pdf)

> **Task**: DragGAN

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Synthesizing visual content that meets users’ needs often requires flexible
and precise controllability of the pose, shape, expression, and layout of the
generated objects. Existing approaches gain controllability of generative
adversarial networks (GANs) via manually annotated training data or a
prior 3D model, which often lack flexibility, precision, and generality. In
this work, we study a powerful yet much less explored way of controlling
GANs, that is, to "drag" any points of the image to precisely reach target
points in a user-interactive manner, as shown in Fig.1. To achieve this, we
propose DragGAN, which consists of two main components: 1) a feature-based motion supervision that drives the handle point to move towards
the target position, and 2) a new point tracking approach that leverages
the discriminative generator features to keep localizing the position of the
handle points. Through DragGAN, anyone can deform an image with precise
control over where pixels go, thus manipulating the pose, shape, expression,and layout of diverse categories such as animals, cars, humans, landscapes,
etc. As these manipulations are performed on the learned generative image
manifold of a GAN, they tend to produce realistic outputs even for challenging scenarios such as hallucinating occluded content and deforming
shapes that consistently follow the object’s rigidity. Both qualitative and
quantitative comparisons demonstrate the advantage of DragGAN over prior
approaches in the tasks of image manipulation and point tracking. We also
showcase the manipulation of real images through GAN inversion.

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/open-mmlab/mmagic/assets/55343765/7c397bd0-fa07-48fe-8a7c-a4022907404b"/>
</div>

## Results and Models

<table><tr>
<b> Gradio Demo of DragGAN StyleGAN2-elephants-512 by MMagic </b>
<td><img src="https://github.com/open-mmlab/mmagic/assets/55343765/08e9a687-0a6e-4d3f-94ec-22c46bd61819" border=0></td>
<td><img src="https://github.com/open-mmlab/mmagic/assets/55343765/6fab1ccd-e190-4cd0-a8d5-0e843f65930b" border=0></td>
</tr></table>

|                        Model                         |      Dataset       |             Comment             | FID50k | Precision50k | Recall50k |                                 Download                                 |
| :--------------------------------------------------: | :----------------: | :-----------------------------: | :----: | :----------: | :-------: | :----------------------------------------------------------------------: |
|   [stylegan2_lion_512x512](./stylegan2_512x512.py)   |   Internet Lions   |     self-distilled StyleGAN     |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-Lions-internet) |
| [stylegan2_elphants_512x512](./stylegan2_512x512.py) | Internet Elephants |     self-distilled StyleGAN     |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-elephants-internet) |
|   [stylegan2_cats_512x512](./stylegan2_512x512.py)   |      Cat AFHQ      |     self-distilled StyleGAN     |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-cat-AFHQ) |
|   [stylegan2_face_512x512](./stylegan2_512x512.py)   |        FFHQ        |     self-distilled StyleGAN     |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-FFHQ) |
|  [stylegan2_horse_256x256](./stylegan2_256x256.py)   |     LSUN-Horse     |     self-distilled StyleGAN     |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-lsun-horses) |
| [stylegan2_dogs_1024x1024](./stylegan2_1024x1024.py) |   Internet Dogs    |     self-distilled StyleGAN     |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-dogs-internet) |
|   [stylegan2_car_512x512](./stylegan2_512x512.py)    |        Car         | transfer from official training |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-car-official) |
|   [stylegan2_cat_256x256](./stylegan2_256x256.py)    |        Cat         | transfer from official training |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-cat-official) |

## Demo

To run DragGAN demo, please follow these two steps:

First, put your checkpoint path in `./checkpoints`, *e.g.* `./checkpoints/stylegan2_lions_512_pytorch_mmagic.pth`. To be specific,

```shell
mkdir checkpoints
cd checkpoints
wget -O stylegan2_lions_512_pytorch_mmagic.pth https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-Lions-internet
```

Then, try on the script:

```shell
python demo/gradio_draggan.py
```

## Citation

```latex
@inproceedings{pan2023drag,
  title={Drag your gan: Interactive point-based manipulation on the generative image manifold},
  author={Pan, Xingang and Tewari, Ayush and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  pages={1--11},
  year={2023}
}
```
