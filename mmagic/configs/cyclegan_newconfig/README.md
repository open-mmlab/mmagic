# CycleGAN (ICCV'2017)

> [CycleGAN: Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)

> **Task**: Image2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G: X \\rightarrow Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F: Y \\rightarrow X and introduce a cycle consistency loss to push F(G(X)) \\approx X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143049598-23c24d98-7a64-4ab3-a9ba-351db6a0a53d.JPG" />
</div>

## Results and Models

<div align="center">
  <b> Results from CycleGAN trained by mmagic</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/114303527-108ed200-9b01-11eb-978c-274392e4d8e0.PNG" width="800"/>
</div>

We use `FID` and `IS` metrics to evaluate the generation performance of CycleGAN.<sup>1</sup>
https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_80k_facades_20210902_165905-5e2c0876.pth
https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_in_1x1_80k_facades_20210902_165905-5e2c0876.pth

|                                 Model                                  |      Dataset      |   FID    |  IS   |                                               Download                                               |
| :--------------------------------------------------------------------: | :---------------: | :------: | :---: | :--------------------------------------------------------------------------------------------------: |
|      [Ours](./cyclegan_lsgan-resnet-in_1xb1-80kiters_facades.py)       |      facades      | 124.8033 | 1.792 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_80k_facades_20210902_165905-5e2c0876.pth) \| [log](https://download.openmmlab.com/mmediting/cyclegan/cyclegan_lsgan_resnet_in_1x1_80k_facades_20210317_160938.log.json) <sup>2</sup> |
|    [Ours](./cyclegan_lsgan-id0-resnet-in_1xb1-80kiters_facades.py)     |    facades-id0    | 125.1694 | 1.905 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_80k_facades_convert-bgr_20210902_164411-d8e72b45.pth) |
|   [Ours](./cyclegan_lsgan-resnet-in_1xb1-250kiters_summer2winter.py)   |   summer2winter   | 83.7177  | 2.771 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165932-fcf08dc1.pth) |
| [Ours](./cyclegan_lsgan-id0-resnet-in_1xb1-250kiters_summer2winter.py) | summer2winter-id0 | 83.1418  | 2.720 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165640-8b825581.pth) |
|   [Ours](./cyclegan_lsgan-resnet-in_1xb1-250kiters_summer2winter.py)   |   winter2summer   | 72.8025  | 3.129 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165932-fcf08dc1.pth) |
| [Ours](./cyclegan_lsgan-id0-resnet-in_1xb1-250kiters_summer2winter.py) | winter2summer-id0 | 73.5001  | 3.107 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_246200_summer2winter_convert-bgr_20210902_165640-8b825581.pth) |
|    [Ours](./cyclegan_lsgan-resnet-in_1xb1-270kiters_horse2zebra.py)    |    horse2zebra    | 64.5225  | 1.418 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_convert-bgr_20210902_170004-a32c733a.pth) |
|  [Ours](./cyclegan_lsgan-id0-resnet-in_1xb1-270kiters_horse2zebra.py)  |  horse2zebra-id0  | 74.7770  | 1.542 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_convert-bgr_20210902_165724-77c9c806.pth) |
|    [Ours](./cyclegan_lsgan-resnet-in_1xb1-270kiters_horse2zebra.py)    |    zebra2horse    | 141.1517 | 3.154 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_resnet_in_1x1_266800_horse2zebra_convert-bgr_20210902_170004-a32c733a.pth) |
|  [Ours](./cyclegan_lsgan-id0-resnet-in_1xb1-270kiters_horse2zebra.py)  |  zebra2horse-id0  | 134.3728 | 3.091 | [model](https://download.openmmlab.com/mmediting/cyclegan/refactor/cyclegan_lsgan_id0_resnet_in_1x1_266800_horse2zebra_convert-bgr_20210902_165724-77c9c806.pth) |

`FID` comparison with official:

<!-- SKIP THIS TABLE -->

| Dataset  |   facades   | facades-id0 | summer2winter | summer2winter-id0 | winter2summer | winter2summer-id0 | horse2zebra | horse2zebra-id0 | zebra2horse | zebra2horse-id0 |  average   |
| :------: | :---------: | :---------: | :-----------: | :---------------: | :-----------: | :---------------: | :---------: | :-------------: | :---------: | :-------------: | :--------: |
| official | **123.626** | **119.726** |  **77.342**   |    **76.773**     |  **72.631**   |      74.239       | **62.111**  |     77.202      | **138.646** |   **137.050**   | **95.935** |
|   ours   |  124.8033   |  125.1694   |    83.7177    |      83.1418      |    72.8025    |    **73.5001**    |   64.5225   |   **74.7770**   |  141.1571   |  **134.3728**   |   97.79    |

`IS` comparison with evaluation:

<!-- SKIP THIS TABLE -->

| Dataset  |  facades  | facades-id0 | summer2winter | summer2winter-id0 | winter2summer | winter2summer-id0 | horse2zebra | horse2zebra-id0 | zebra2horse | zebra2horse-id0 |  average  |
| :------: | :-------: | :---------: | :-----------: | :---------------: | :-----------: | :---------------: | :---------: | :-------------: | :---------: | :-------------: | :-------: |
| official |   1.638   |    1.697    |     2.762     |     **2.750**     |   **3.293**   |     **3.110**     |    1.375    |    **1.584**    |  **3.186**  |      3.047      |   2.444   |
|   ours   | **1.792** |  **1.905**  |   **2.771**   |       2.720       |     3.129     |       3.107       |  **1.418**  |      1.542      |    3.154    |    **3.091**    | **2.462** |

Note:

1. With a larger identity loss, the image-to-image translation becomes more conservative, which makes less changes. The original authors did not say what is the best weight for identity loss. Thus, in addition to the default setting, we also set the weight of identity loss to 0 (denoting `id0`) to make a more comprehensive comparison.
2. This is the training log before refactoring. Updated logs will be released soon.

## Citation

```latex
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017},
  url={https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html},
}
```
