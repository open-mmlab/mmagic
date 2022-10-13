# Pix2Pix (CVPR'2017)

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1611.07004">Pix2Pix (CVPR'2017)</a></summary>

```bibtex
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```

</details>

<br/>

我们使用 `FID` 和 `IS` 指标来评估 pix2pix 的生成表现。

|                                            算法                                            |     FID     |    IS     |                                            下载                                            |
| :----------------------------------------------------------------------------------------: | :---------: | :-------: | :----------------------------------------------------------------------------------------: |
|                                        官方 facades                                        | **119.135** |   1.650   |                                             -                                              |
|  [复现 facades](/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py)  |   127.792   | **1.745** | [模型](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_facades/pix2pix_vanilla_unet_bn_1x1_80k_facades_20200524-6206de67.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_facades/pix2pix_vanilla_unet_bn_1x1_80k_facades_20200524_185039.log.json) |
|                                       官方 maps-a2b                                        |   149.731   |   2.529   |                                             -                                              |
| [复现 maps-a2b](/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps.py) | **118.552** | **2.689** | [模型](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_maps_a2b/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps_20200524-b29c4538.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_maps_a2b/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps_20200524_191918.log.json) |
|                                       官方 maps-b2a                                        |   102.072   | **3.552** |                                             -                                              |
| [复现 maps-b2a](/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps.py) | **92.798**  |   3.473   | [模型](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_maps_b2a/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps_20200524-17882ec8.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_maps_b2a/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps_20200524_192641.log.json) |
|                                      官方 edges2shoes                                      | **75.774**  | **2.766** |                                             -                                              |
| [复现 edges2shoes](/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes.py) |   85.413    |   2.747   | [模型](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_edges2shoes_wo_jitter_flip/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_20200524-b35fa9c0.pth) \| [日志](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_edges2shoes_wo_jitter_flip/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_20200524_193117.log.json) |
|                                         官方平均值                                         |   111.678   |   2.624   |                                             -                                              |
|                                         复现平均值                                         | **106.139** | **2.664** |                                             -                                              |

注：我们严格遵守[论文](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)第3.3节中的设置：

"*At inference time, we run the generator net in exactly
the same manner as during the training phase. This differs
from the usual protocol in that we apply dropout at test time,
and we apply batch normalization using the statistics of
the test batch, rather than aggregated statistics of the training batch.*"

即使用 `model.train()` 模式，因此可能会导致每次推理结果略有不同。
