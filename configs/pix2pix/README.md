# Pix2Pix (CVPR'2017)

> [Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html)

> **Task**: Image2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Moreover, since the release of the pix2pix software associated with this paper, hundreds of twitter users have posted their own artistic experiments using our system. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without handengineering our loss functions either.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143053385-1b03356d-43df-423b-88b2-7a82b73d2edd.JPG"/>
</div>

## Results and Models

<div align="center">
  <b> Results from Pix2Pix trained by mmagic</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/114269080-4ff0ec00-9a37-11eb-92c4-1525864e0307.PNG" width="800"/>
</div>
We use `FID` and `IS` metrics to evaluate the generation performance of pix2pix.<sup>1</sup>

|                                     Model                                      |   Dataset   |   FID    |  IS   |                                              Download                                              |
| :----------------------------------------------------------------------------: | :---------: | :------: | :---: | :------------------------------------------------------------------------------------------------: |
|           [Ours](./pix2pix_vanilla-unet-bn_1xb1-80kiters_facades.py)           |   facades   | 124.9773 | 1.620 | [model](https://download.openmmlab.com/mmediting/pix2pix/refactor/pix2pix_vanilla_unet_bn_1x1_80k_facades_20210902_170442-c0958d50.pth)   \| [log](https://download.openmmlab.com/mmediting/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades_20210317_172625.log.json)<sup>2</sup> |
|        [Ours](./pix2pix_vanilla-unet-bn_1xb1-220kiters_aerial2maps.py)         | aerial2maps | 122.5856 | 3.137 | [model](https://download.openmmlab.com/mmediting/pix2pix/refactor/pix2pix_vanilla_unet_bn_a2b_1x1_219200_maps_convert-bgr_20210902_170729-59a31517.pth) |
|        [Ours](./pix2pix_vanilla-unet-bn_1xb1-220kiters_maps2aerial.py)         | maps2aerial | 88.4635  | 3.310 | [model](https://download.openmmlab.com/mmediting/pix2pix/refactor/pix2pix_vanilla_unet_bn_b2a_1x1_219200_maps_convert-bgr_20210902_170814-6d2eac4a.pth) |
| [Ours](./pix2pix_vanilla-unet-bn_wo-jitter-flip-1xb4-190kiters_edges2shoes.py) | edges2shoes | 84.3750  | 2.815 | [model](https://download.openmmlab.com/mmediting/pix2pix/refactor/pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_convert-bgr_20210902_170902-0c828552.pth) |

`FID` comparison with official:

<!-- SKIP THIS TABLE -->

| Dataset  |   facades   | aerial2maps  | maps2aerial | edges2shoes |   average    |
| :------: | :---------: | :----------: | :---------: | :---------: | :----------: |
| official | **119.135** |   149.731    |   102.072   | **75.774**  |   111.678    |
|   ours   |  124.9773   | **122.5856** | **88.4635** |   84.3750   | **105.1003** |

`IS` comparison with official:

<!-- SKIP THIS TABLE -->

| Dataset  |  facades  | aerial2maps | maps2aerial | edges2shoes |  average   |
| :------: | :-------: | :---------: | :---------: | :---------: | :--------: |
| official | **1.650** |    2.529    |  **3.552**  |    2.766    |   2.624    |
|   ours   |   1.620   |  **3.137**  |    3.310    |  **2.815**  | **2.7205** |

Note:

1. we strictly follow the [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf) setting in Section 3.3: "*At inference time, we run the generator net in exactly
   the same manner as during the training phase. This differs
   from the usual protocol in that we apply dropout at test time,
   and we apply batch normalization using the statistics of
   the test batch, rather than aggregated statistics of the training batch.*" (i.e., use model.train() mode), thus may lead to slightly different inference results every time.
2. This is the training log before refactoring. Updated logs will be released soon.

## Citation

```latex
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017},
  url={https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html},
}
```
