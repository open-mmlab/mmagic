# DragGAN (SIGGRAPH'2023)

> [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://arxiv.org/pdf/2305.10973.pdf)

> **Task**: DragGAN

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

要合成满足用户需求的视觉内容，通常需要灵活、精确地控制人物的姿势、形状、表情和布局。
和精确地控制生成对象的姿势、形状、表情和布局。
生成对象。现有的方法是通过人工标注训练数据来获得生成对抗网络（GANs）的可控性。
生成式对抗网络 (GAN) 的可控性是通过人工标注的训练数据或预先的 3D 模型来实现的。
生成式对抗网络（GAN）的可控性是通过人工标注的训练数据或事先建立的三维模型来实现的，而这些方法往往缺乏灵活性、精确性和通用性。在中，我们研究了一种功能强大但探索较少的生成式对抗网络（GAN）控制方法。
即 "拖动 "图像中的任意点，以用户互动的方式精确到达目标点。
如图 1 所示。为此，我们
DragGAN 由两个主要部分组成： 1) 基于特征的运动监督，驱动手柄点向目标位置移动；以及目标位置，以及 2) 一种新的点跟踪方法，利用的新点跟踪方法。
手柄点的位置。通过 DragGAN，任何人都可以对图像进行变形，并精确控制像素的移动位置。
通过 DragGAN，任何人都可以精确控制像素的位置，从而改变图像的姿态、形状、表情和布局，如动物、汽车、人类、风景等、
等等。由于这些操作是在 GAN 的学习生成图像流形上进行的流形上进行，因此即使是在具有挑战性的情况下，它们也能产生逼真的输出，例如幻觉遮挡内容和变形形状始终遵循物体的刚性。定性和定量比较都表明，在图像处理和点跟踪任务中，DragGAN
在图像处理和点跟踪任务中的优势。我们还展示了通过 GAN 反演对真实图像的操作。

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/open-mmlab/mmagic/assets/55343765/7c397bd0-fa07-48fe-8a7c-a4022907404b"/>
</div>

## 结果和模型

<table><tr>
<b> MMagic下实现的DragGAN（StyleGAN2-elephants-512） Gradio示例 </b>
<td><img src="https://github.com/open-mmlab/mmagic/assets/55343765/08e9a687-0a6e-4d3f-94ec-22c46bd61819" border=0></td>
<td><img src="https://github.com/open-mmlab/mmagic/assets/55343765/6fab1ccd-e190-4cd0-a8d5-0e843f65930b" border=0></td>
</tr></table>

|                         模型                         |       数据集       |      评论      | FID50k | Precision50k | Recall50k |                                       权重下载链接                                        |
| :--------------------------------------------------: | :----------------: | :------------: | :----: | :----------: | :-------: | :---------------------------------------------------------------------------------------: |
|   [stylegan2_lion_512x512](./stylegan2_512x512.py)   |   Internet Lions   | 自蒸馏StyleGAN |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-Lions-internet) |
| [stylegan2_elphants_512x512](./stylegan2_512x512.py) | Internet Elephants | 自蒸馏StyleGAN |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-elephants-internet) |
|   [stylegan2_cats_512x512](./stylegan2_512x512.py)   |      Cat AFHQ      | 自蒸馏StyleGAN |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-cat-AFHQ) |
|   [stylegan2_face_512x512](./stylegan2_512x512.py)   |        FFHQ        | 自蒸馏StyleGAN |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-FFHQ) |
|  [stylegan2_horse_256x256](./stylegan2_256x256.py)   |     LSUN-Horse     | 自蒸馏StyleGAN |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-lsun-horses) |
| [stylegan2_dogs_1024x1024](./stylegan2_1024x1024.py) |   Internet Dogs    | 自蒸馏StyleGAN |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-dogs-internet) |
|   [stylegan2_car_512x512](./stylegan2_512x512.py)    |        Car         |    官方训练    |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-car-official) |
|   [stylegan2_cat_256x256](./stylegan2_256x256.py)    |        Cat         |    官方训练    |  0.0   |     0.0      |    0.0    | [model](https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-cat-official) |

## 演示

为了使用DragGAN演示, 请执行以下两步:

首先，把模型文件放在 `./checkpoints`下, 比如 `./checkpoints/stylegan2_lions_512_pytorch_mmagic.pth`. 具体来说,

```shell
mkdir checkpoints
cd checkpoints
wget -O stylegan2_lions_512_pytorch_mmagic.pth https://download.openxlab.org.cn/models/qsun1/DragGAN-StyleGAN2-checkpoint/weight//StyleGAN2-Lions-internet
```

然后，执行下面的脚本:

```shell
python demo/gradio_draggan.py
```

## 引用

```latex
@inproceedings{pan2023drag,
  title={Drag your gan: Interactive point-based manipulation on the generative image manifold},
  author={Pan, Xingang and Tewari, Ayush and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  pages={1--11},
  year={2023}
}
```
