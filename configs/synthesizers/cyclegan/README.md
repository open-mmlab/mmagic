# CycleGAN (ICCV'2017)

> [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144200449-cc2777da-3112-4024-aaa6-c6be5c8220bc.png" width="400"/>
</div >

## Results and models

We use `FID` and `IS` metrics to evaluate the generation performance of CycleGAN.

|                                          Method                                           |     FID     |    IS     |                                          Download                                           |
| :---------------------------------------------------------------------------------------: | :---------: | :-------: | :-----------------------------------------------------------------------------------------: |
|                                     official facades                                      |   123.626   | **1.638** |                                              -                                              |
| [ours facades](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_80k_facades.py) | **118.297** |   1.584   | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_facades/cyclegan_lsgan_resnet_in_1x1_80k_facades_20200524-0b877c2a.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_facades/cyclegan_lsgan_resnet_in_1x1_80k_facades_20200524_211816.log.json) |
|                                   official facades-id0                                    | **119.726** |   1.697   |                                              -                                              |
| [ours facades-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_80k_facades.py) |   126.316   | **1.957** | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_facades_id0/cyclegan_lsgan_id0_resnet_in_1x1_80k_facades_20200524-438aa074.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_facades_id0/cyclegan_lsgan_id0_resnet_in_1x1_80k_facades_20200524_212548.log.json) |
|                                  official summer2winter                                   |   77.342    |   2.762   |                                              -                                              |
| [ours summer2winter](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter.py) | **76.959**  | **2.768** | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_20200524-0baeaff6.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_20200524_214809.log.json) |
|                                  official winter2summer                                   | **72.631**  | **3.293** |                                              -                                              |
| [ours winter2summer](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter.py) |   72.803    |   3.069   | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_20200524-0baeaff6.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_20200524_214809.log.json) |
|                                official summer2winter-id0                                 |   76.773    | **2.750** |                                              -                                              |
| [ours summer2winter-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter.py) | **76.018**  |   2.735   | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter_id0/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_20200524-f280ecdd.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter_id0/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_20200524_215511.log.json) |
|                                official winter2summer-id0                                 |   74.239    |   3.110   |                                              -                                              |
| [ours winter2summer-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter.py) | **73.498**  | **3.130** | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter_id0/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_20200524-f280ecdd.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter_id0/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_20200524_215511.log.json) |
|                                   official horse2zebra                                    | **62.111**  |   1.375   |                                              -                                              |
| [ours horse2zebra](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra.py) |   63.810    | **1.430** | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_20200524-1b3d5d3a.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_20200524_220040.log.json) |
|                                 official horse2zebra-id0                                  |   77.202    | **1.584** |                                              -                                              |
| [ours horse2zebra-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra.py) | **71.675**  |   1.542   | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra_id0/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_20200524-470fb8da.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra_id0/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_20200524_220655.log.json) |
|                                   official horse2zebra                                    | **138.646** | **3.186** |                                              -                                              |
| [ours zebra2horse](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra.py) |   139.279   |   3.093   | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_20200524-1b3d5d3a.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_20200524_220040.log.json) |
|                                 official horse2zebra-id0                                  |   137.050   | **3.047** |                                              -                                              |
| [ours zebra2horse-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra.py) | **132.369** |   2.958   | [model](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra_id0/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_20200524-470fb8da.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra_id0/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_20200524_220655.log.json) |
|                                     official average                                      |   95.935    | **2.444** |                                              -                                              |
|                                       ours average                                        | **95.102**  |   2.427   |                                              -                                              |

Note: With a larger identity loss, the image-to-image translation becomes more conservative, which makes less changes. The original authors did not say what is the best weight for identity loss. Thus, in addition to the default setting, we also set the weight of identity loss to 0 (denoting `id0` ) to make a more comprehensive comparison.

## Citation

```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}
```
