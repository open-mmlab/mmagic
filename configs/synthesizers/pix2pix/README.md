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

We use `FID` and `IS` metrics to evaluate the generation performance of pix2pix.

| Method | FID | IS  | Download |
| :----: | :-: | :-: | :------: |
| official facades | **119.135** | 1.650 | - |
| [ours facades](/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py) | 127.792 | **1.745** | [model](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_facades/pix2pix_vanilla_unet_bn_1x1_80k_facades_20200524-6206de67.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_facades/pix2pix_vanilla_unet_bn_1x1_80k_facades_20200524_185039.log.json) |
| official maps-a2b | 149.731 | 2.529 | - |
| [ours maps-a2b](/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps.py) | **118.552** | **2.689** | [model](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_maps_a2b/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps_20200524-b29c4538.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_maps_a2b/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps_20200524_191918.log.json) |
| official maps-b2a | 102.072 | **3.552** | - |
| [ours maps-b2a](/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps.py) | **92.798** | 3.473 | [model](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_maps_b2a/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps_20200524-17882ec8.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_maps_b2a/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps_20200524_192641.log.json) |
| official edges2shoes | **75.774** | **2.766** | - |
| [ours edges2shoes](/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes.py) | 85.413 | 2.747 | [model](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_edges2shoes_wo_jitter_flip/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_20200524-b35fa9c0.pth) \| [log](https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_edges2shoes_wo_jitter_flip/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_20200524_193117.log.json) |
| official average | 111.678 | 2.624 | - |
| ours average | **106.139** | **2.664** | - |

Note: we strictly follow the [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf) setting in Section 3.3: "*At inference time, we run the generator net in exactly
the same manner as during the training phase. This differs
from the usual protocol in that we apply dropout at test time,
and we apply batch normalization using the statistics of
the test batch, rather than aggregated statistics of the training batch.*" (i.e., use model.train() mode), thus may lead to slightly different inference results every time.
