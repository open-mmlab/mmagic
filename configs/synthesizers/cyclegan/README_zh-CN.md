# CycleGAN (ICCV'2017)

<!-- [ALGORITHM] -->

<details>
<summary align="right">CycleGAN (ICCV'2017)</summary>

```bibtex
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017}
}
```

</details>

<br/>

我们使用“FID”和“IS”指标来评估 CycleGAN 的生成性能。

|                                            算法                                            |     FID     |    IS     |                                            下载                                            |
| :----------------------------------------------------------------------------------------: | :---------: | :-------: | :----------------------------------------------------------------------------------------: |
|                                        官方 facades                                        |   123.626   | **1.638** |                                             -                                              |
| [复现 facades](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_80k_facades.py) | **118.297** |   1.584   | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_facades/cyclegan_lsgan_resnet_in_1x1_80k_facades_20200524-0b877c2a.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_facades/cyclegan_lsgan_resnet_in_1x1_80k_facades_20200524_211816.log.json) |
|                                      官方 facades-id0                                      | **119.726** |   1.697   |                                             -                                              |
| [复现 facades-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_80k_facades.py) |   126.316   | **1.957** | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_facades_id0/cyclegan_lsgan_id0_resnet_in_1x1_80k_facades_20200524-438aa074.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_facades_id0/cyclegan_lsgan_id0_resnet_in_1x1_80k_facades_20200524_212548.log.json) |
|                                     官方 summer2winter                                     |   77.342    |   2.762   |                                             -                                              |
| [复现 summer2winter](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter.py) | **76.959**  | **2.768** | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_20200524-0baeaff6.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_20200524_214809.log.json) |
|                                     官方 winter2summer                                     | **72.631**  | **3.293** |                                             -                                              |
| [复现 winter2summer](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter.py) |   72.803    |   3.069   | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_20200524-0baeaff6.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_20200524_214809.log.json) |
|                                   官方 summer2winter-id0                                   |   76.773    | **2.750** |                                             -                                              |
| [复现 summer2winter-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter.py) | **76.018**  |   2.735   | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter_id0/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_20200524-f280ecdd.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter_id0/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_20200524_215511.log.json) |
|                                   官方 winter2summer-id0                                   |   74.239    |   3.110   |                                             -                                              |
| [复现 winter2summer-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter.py) | **73.498**  | **3.130** | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter_id0/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_20200524-f280ecdd.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_summer2winter_id0/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_20200524_215511.log.json) |
|                                      官方 horse2zebra                                      | **62.111**  |   1.375   |                                             -                                              |
| [复现 horse2zebra](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra.py) |   63.810    | **1.430** | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_20200524-1b3d5d3a.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_20200524_220040.log.json) |
|                                    官方 horse2zebra-id0                                    |   77.202    | **1.584** |                                             -                                              |
| [复现 horse2zebra-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra.py) | **71.675**  |   1.542   | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra_id0/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_20200524-470fb8da.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra_id0/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_20200524_220655.log.json) |
|                                      官方 horse2zebra                                      | **138.646** | **3.186** |                                             -                                              |
| [复现 zebra2horse](/configs/synthesizers/cyclegan/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra.py) |   139.279   |   3.093   | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_20200524-1b3d5d3a.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_20200524_220040.log.json) |
|                                    官方 horse2zebra-id0                                    |   137.050   | **3.047** |                                             -                                              |
| [复现 zebra2horse-id0](/configs/synthesizers/cyclegan/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra.py) | **132.369** |   2.958   | [模型](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra_id0/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_20200524-470fb8da.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/cyclegan/cyclegan_horse2zebra_id0/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_20200524_220655.log.json) |
|                                         官方平均值                                         |   95.935    | **2.444** |                                             -                                              |
|                                         复现平均值                                         | **95.102**  |   2.427   |                                             -                                              |

注：随着更大的身份损失，图像到图像的转换会变得更加保守，这导致转换较少。原作者没有提及身份损失的最佳权重。因此，除了默认设置之外，我们还将身份损失的权重设置为 0（表示 `id0`）以进行更全面的比较。
