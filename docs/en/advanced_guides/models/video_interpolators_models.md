# Frame-Interpolation Models

## AOT-GAN (TVCG'2021)

> [AOT-GAN: Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/pdf/2104.01431.pdf)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

State-of-the-art image inpainting approaches can suffer from generating distorted structures and blurry textures in high-resolution images (e.g., 512x512). The challenges mainly drive from (1) image content reasoning from distant contexts, and (2) fine-grained texture synthesis for a large missing region. To overcome these two challenges, we propose an enhanced GAN-based model, named Aggregated COntextual-Transformation GAN (AOT-GAN), for high-resolution image inpainting. Specifically, to enhance context reasoning, we construct the generator of AOT-GAN by stacking multiple layers of a proposed AOT block. The AOT blocks aggregate contextual transformations from various receptive fields, allowing to capture both informative distant image contexts and rich patterns of interest for context reasoning. For improving texture synthesis, we enhance the discriminator of AOT-GAN by training it with a tailored mask-prediction task. Such a training objective forces the discriminator to distinguish the detailed appearances of real and synthesized patches, and in turn, facilitates the generator to synthesize clear textures. Extensive comparisons on Places2, the most challenging benchmark with 1.8 million high-resolution images of 365 complex scenes, show that our model outperforms the state-of-the-art by a significant margin in terms of FID with 38.60% relative improvement. A user study including more than 30 subjects further validates the superiority of AOT-GAN. We further evaluate the proposed AOT-GAN in practical applications, e.g., logo removal, face editing, and object removal. Results show that our model achieves promising completions in the real world. We release code and models in [this https URL](https://github.com/researchmm/AOT-GAN-for-Inpainting).

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12756472/169230414-3ca7fb6b-cf2a-401f-8696-71df75a08c32.png"/>
</div >

### Results and models

**Places365-Challenge**

|                                                    Method                                                     |     Mask Type      | Resolution | Train Iters |   Test Set    | l1 error | PSNR  | SSIM  |        GPU Info         |                                                                                                                                          Download                                                                                                                                           |
| :-----------------------------------------------------------------------------------------------------------: | :----------------: | :--------: | :---------: | :-----------: | :------: | :---: | :---: | :---------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [AOT-GAN](https://github.com/open-mmlab/mmediting/tree/master/configs/AOT-GAN/AOT-GAN_512x512_4x12_places.py) | free-form (50-60%) |  512x512   |    500k     | Places365-val |   7.07   | 19.01 | 0.682 | 4 (GeForce GTX 1080 Ti) | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmediting/inpainting/aot_gan/AOT-GAN_512x512_4x12_places_20220509-6641441b.pth) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmediting/inpainting/aot_gan/AOT-GAN_512x512_4x12_places_20220509-6641441b.json) |

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

### Citation

```bibtex
@inproceedings{yan2021agg,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Aggregated Contextual Transformations for High-Resolution Image Inpainting},
  booktitle = {Arxiv},
  pages={-},
  year = {2020}
}
```

## BasicVSR++ (CVPR'2022)

> [BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment](https://arxiv.org/abs/2104.13371)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

A recurrent structure is a popular framework choice for the task of video super-resolution. The state-of-the-art method BasicVSR adopts bidirectional propagation with feature alignment to effectively exploit information from the entire input video. In this study, we redesign BasicVSR by proposing second-order grid propagation and flow-guided deformable alignment. We show that by empowering the recurrent framework with the enhanced propagation and alignment, one can exploit spatiotemporal information across misaligned video frames more effectively. The new components lead to an improved performance under a similar computational constraint. In particular, our model BasicVSR++ surpasses BasicVSR by 0.82 dB in PSNR with similar number of parameters. In addition to video super-resolution, BasicVSR++ generalizes well to other video restoration tasks such as compressed video enhancement. In NTIRE 2021, BasicVSR++ obtains three champions and one runner-up in the Video Super-Resolution and Compressed Video Enhancement Challenges. Codes and models will be released to MMEditing.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144017685-9354df55-aa6d-445f-a946-116f0d6c38d7.png" width="400"/>
</div >

### Results and models

The pretrained weights of SPyNet can be found [here](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth).

|                                                                                    Method                                                                                     | REDS4 (BIx4) PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4) PSNR/SSIM (Y) | Vid4 (BIx4) PSNR/SSIM (Y) | UDM10 (BDx4) PSNR/SSIM (Y) | Vimeo-90K-T (BDx4) PSNR/SSIM (Y) | Vid4 (BDx4) PSNR/SSIM (Y) |         GPU Info         |                                                                                                                                               Download                                                                                                                                                |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------: | :------------------------------: | :-----------------------: | :------------------------: | :------------------------------: | :-----------------------: | :----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [basicvsr_plusplus_c64n7_8x1_600k_reds4](https://github.com/open-mmlab/mmediting/tree/master/configs/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py)       |      **32.3855/0.9069**      |          36.4445/0.9411          |      27.7674/0.8444       |       34.6868/0.9417       |          34.0372/0.9244          |      24.6209/0.7540       | 8 (Tesla V100-PCIE-32GB) |       [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217_113115.log.json)       |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi](https://github.com/open-mmlab/mmediting/tree/master/configs/basicvsr_plusplus/basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bi.py) |        31.0126/0.8804        |        **37.7864/0.9500**        |    **27.7882/0.8401**     |       33.1211/0.9270       |          33.8972/0.9195          |      23.6086/0.7033       | 4 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305_141254.log.json) |
| [basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd](https://github.com/open-mmlab/mmediting/tree/master/configs/basicvsr_plusplus/basicvsr_plusplus_c64n7_4x2_300k_vimeo90k_bd.py) |        29.2041/0.8528        |          34.7248/0.9351          |      26.4377/0.8074       |     **40.7216/0.9722**     |        **38.2054/0.9550**        |    **29.0400/0.8753**     | 4 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305_140921.log.json) |

<details>
<summary align="left">NTIRE 2021 checkpoints</summary>

Note that the following models are finetuned from smaller models. The training schemes of these models will be released when MMEditing reaches 5k stars. We provide the pre-trained models here.

[NTIRE 2021 Video Super-Resolution](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 1](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 2](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track2_20210314-eeae05e6.pth)

[NTIRE 2021 Quality Enhancement of Compressed Video - Track 3](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track3_20210304-6daf4a40.pth)

</details>

### Citation

```bibtex
@InProceedings{chan2022basicvsrplusplus,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2022}
}
```

## BasicVSR (CVPR'2021)

> [BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond](https://arxiv.org/abs/2012.02181)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Video super-resolution (VSR) approaches tend to have more components than the image counterparts as they need to exploit the additional temporal dimension. Complex designs are not uncommon. In this study, we wish to untangle the knots and reconsider some most essential components for VSR guided by four basic functionalities, i.e., Propagation, Alignment, Aggregation, and Upsampling. By reusing some existing components added with minimal redesigns, we show a succinct pipeline, BasicVSR, that achieves appealing improvements in terms of speed and restoration quality in comparison to many state-of-the-art algorithms. We conduct systematic analysis to explain how such gain can be obtained and discuss the pitfalls. We further show the extensibility of BasicVSR by presenting an information-refill mechanism and a coupled propagation scheme to facilitate information aggregation. The BasicVSR and its extension, IconVSR, can serve as strong baselines for future VSR approaches.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144011085-fdded077-24de-468b-826e-5f82716219a5.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels for REDS4 and Y channel for others. The metrics are `PSNR` / `SSIM` .
The pretrained weights of SPyNet can be found [here](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth).

|                                                        Method                                                        | REDS4 (BIx4)<br>PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4)<br>PSNR/SSIM (Y) | Vid4 (BIx4)<br>PSNR/SSIM (Y) | UDM10 (BDx4)<br>PSNR/SSIM (Y) | Vimeo-90K-T (BDx4)<br>PSNR/SSIM (Y) | Vid4 (BDx4)<br>PSNR/SSIM (Y) |         GPU Info         |                                                                                                              Download                                                                                                               |
| :------------------------------------------------------------------------------------------------------------------: | :-----------------------------: | :---------------------------------: | :--------------------------: | :---------------------------: | :---------------------------------: | :--------------------------: | :----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [basicvsr_reds4](https://github.com/open-mmlab/mmediting/tree/master/configs/basicvsr/basicvsr_reds4.py)       |       **31.4170/0.8909**        |           36.2848/0.9395            |        27.2694/0.8318        |        33.4478/0.9306         |           34.4700/0.9286            |        24.4541/0.7455        | 2 (Tesla V100-PCIE-32GB) |       [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20210409_092646.log.json)       |
| [basicvsr_vimeo90k_bi](https://github.com/open-mmlab/mmediting/tree/master/configs/basicvsr/basicvsr_vimeo90k_bi.py) |         30.3128/0.8660          |         **37.2026/0.9451**          |      **27.2755/0.8248**      |        34.5554/0.9434         |           34.8097/0.9316            |        25.0517/0.7636        | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409-d2d8f760.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409_132702.log.json) |
| [basicvsr_vimeo90k_bd](https://github.com/open-mmlab/mmediting/tree/master/configs/basicvsr/basicvsr_vimeo90k_bd.py) |         29.0376/0.8481          |           34.6427/0.9335            |        26.2708/0.8022        |      **39.9953/0.9695**       |         **37.5501/0.9499**          |      **27.9791/0.8556**      | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409-0154dd64.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409_132740.log.json) |

### Citation

```bibtex
@InProceedings{chan2021basicvsr,
  author = {Chan, Kelvin CK and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title = {BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

## CAIN (AAAI'2020)

> [Channel Attention Is All You Need for Video Frame Interpolation](https://aaai.org/ojs/index.php/AAAI/article/view/6693/6547)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Prevailing video frame interpolation techniques rely heavily on optical flow estimation and require additional model complexity and computational cost; it is also susceptible to error propagation in challenging scenarios with large motion and heavy occlusion. To alleviate the limitation, we propose a simple but effective deep neural network for video frame interpolation, which is end-to-end trainable and is free from a motion estimation network component. Our algorithm employs a special feature reshaping operation, referred to as PixelShuffle, with a channel attention, which replaces the optical flow computation module. The main idea behind the design is to distribute the information in a feature map into multiple channels and extract motion information by attending the channels for pixel-level frame synthesis. The model given by this principle turns out to be effective in the presence of challenging motion and occlusion. We construct a comprehensive evaluation benchmark and demonstrate that the proposed approach achieves outstanding performance compared to the existing models with a component for optical flow computation.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/56712176/149734064-1da0cebf-6953-4106-a29a-43acd7386a80.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .
The learning rate adjustment strategy is `Step LR scheduler with min_lr clipping`.

|                                                                Method                                                                | vimeo-90k-triplet |         GPU Info         |                                                                                                                              Download                                                                                                                              |
| :----------------------------------------------------------------------------------------------------------------------------------: | :---------------: | :----------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [cain_b5_g1b32_vimeo90k_triplet](https://github.com/open-mmlab/mmediting/tree/master/configs/cain/cain_b5_g1b32_vimeo90k_triplet.py) | 34.6010 / 0.9578  | 1 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.pth)/[log](https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.log.json) |

### Citation

```bibtex
@inproceedings{choi2020channel,
  title={Channel attention is all you need for video frame interpolation},
  author={Choi, Myungsub and Kim, Heewon and Han, Bohyung and Xu, Ning and Lee, Kyoung Mu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={10663--10671},
  year={2020}
}
```

## DeepFillv1 (CVPR'2018)

> [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Recent deep learning based approaches have shown promising results for the challenging task of inpainting large missing regions in an image. These methods can generate visually plausible image structures and textures, but often create distorted structures or blurry textures inconsistent with surrounding areas. This is mainly due to ineffectiveness of convolutional neural networks in explicitly borrowing or copying information from distant spatial locations. On the other hand, traditional texture and patch synthesis approaches are particularly suitable when it needs to borrow textures from the surrounding regions. Motivated by these observations, we propose a new deep generative model-based approach which can not only synthesize novel image structures but also explicitly utilize surrounding image features as references during network training to make better predictions. The model is a feed-forward, fully convolutional neural network which can process images with multiple holes at arbitrary locations and with variable sizes during the test time. Experiments on multiple datasets including faces (CelebA, CelebA-HQ), textures (DTD) and natural images (ImageNet, Places2) demonstrate that our proposed approach generates higher-quality inpainting results than existing ones.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144174665-9675931f-e448-4475-a659-99b65e7d4a64.png" width="400"/>
</div >

### Results and models

**Places365-Challenge**

|                                                        Method                                                         |  Mask Type  | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  | GPU Info |                                                                                                                           Download                                                                                                                            |
| :-------------------------------------------------------------------------------------------------------------------: | :---------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DeepFillv1](https://github.com/open-mmlab/mmediting/tree/master/configs/deepfillv1/deepfillv1_256x256_8x2_places.py) | square bbox |  256x256   |    3500k    | Places365-val |  11.019  | 23.429 | 0.862 |    8     | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.log.json) |

**CelebA-HQ**

|                                                        Method                                                         |  Mask Type  | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  | GPU Info |                                                                                                                           Download                                                                                                                            |
| :-------------------------------------------------------------------------------------------------------------------: | :---------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DeepFillv1](https://github.com/open-mmlab/mmediting/tree/master/configs/deepfillv1/deepfillv1_256x256_4x4_celeba.py) | square bbox |  256x256   |    1500k    | CelebA-val |  6.677   | 26.878 | 0.911 |    4     | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.log.json) |

### Citation

```bibtex
@inproceedings{yu2018generative,
  title={Generative image inpainting with contextual attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5505--5514},
  year={2018}
}
```

## DeepFillv2 (CVPR'2019)

> [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We present a generative image inpainting system to complete images with free-form mask and guidance. The system is based on gated convolutions learned from millions of images without additional labelling efforts. The proposed gated convolution solves the issue of vanilla convolution that treats all input pixels as valid ones, generalizes partial convolution by providing a learnable dynamic feature selection mechanism for each channel at each spatial location across all layers. Moreover, as free-form masks may appear anywhere in images with any shape, global and local GANs designed for a single rectangular mask are not applicable. Thus, we also present a patch-based GAN loss, named SN-PatchGAN, by applying spectral-normalized discriminator on dense image patches. SN-PatchGAN is simple in formulation, fast and stable in training. Results on automatic image inpainting and user-guided extension demonstrate that our system generates higher-quality and more flexible results than previous methods. Our system helps user quickly remove distracting objects, modify image layouts, clear watermarks and edit faces.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175160-75473789-924f-490b-ab25-4c4f252fa55f.png" width="400"/>
</div >

### Results and models

**Places365-Challenge**

|                                                        Method                                                         | Mask Type | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  | GPU Info |                                                                                                                           Download                                                                                                                            |
| :-------------------------------------------------------------------------------------------------------------------: | :-------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DeepFillv2](https://github.com/open-mmlab/mmediting/tree/master/configs/deepfillv2/deepfillv2_256x256_8x2_places.py) | free-form |  256x256   |    100k     | Places365-val |  8.635   | 22.398 | 0.815 |    8     | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.log.json) |

**CelebA-HQ**

|                                                        Method                                                         | Mask Type | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  | GPU Info |                                                                                                                           Download                                                                                                                            |
| :-------------------------------------------------------------------------------------------------------------------: | :-------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DeepFillv2](https://github.com/open-mmlab/mmediting/tree/master/configs/deepfillv2/deepfillv2_256x256_8x2_celeba.py) | free-form |  256x256   |     20k     | CelebA-val |  5.411   | 25.721 | 0.871 |    8     | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.log.json) |

### Citation

```bibtex
@inproceedings{yu2019free,
  title={Free-form image inpainting with gated convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4471--4480},
  year={2019}
}
```

## DIC (CVPR'2020)

> [Deep Face Super-Resolution with Iterative Collaboration between Attentive Recovery and Landmark Estimation](https://arxiv.org/abs/2003.13063)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Recent works based on deep learning and facial priors have succeeded in super-resolving severely degraded facial images. However, the prior knowledge is not fully exploited in existing methods, since facial priors such as landmark and component maps are always estimated by low-resolution or coarsely super-resolved images, which may be inaccurate and thus affect the recovery performance. In this paper, we propose a deep face super-resolution (FSR) method with iterative collaboration between two recurrent networks which focus on facial image recovery and landmark estimation respectively. In each recurrent step, the recovery branch utilizes the prior knowledge of landmarks to yield higher-quality images which facilitate more accurate landmark estimation in turn. Therefore, the iterative information interaction between two processes boosts the performance of each other progressively. Moreover, a new attentive fusion module is designed to strengthen the guidance of landmark maps, where facial components are generated individually and aggregated attentively for better restoration. Quantitative and qualitative experimental results show the proposed method significantly outperforms state-of-the-art FSR methods in recovering high-quality face images.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144017838-63e31123-1b59-4743-86bb-737bd32a9209.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

In the log data of `dic_gan_x8c48b6_g4_150k_CelebAHQ`, DICGAN is verified on the first 9 pictures of the test set of CelebA-HQ, so `PSNR/SSIM` shown in the follow table is different from the log data.

`GPU Info`: GPU information during training.

|                                                                 Method                                                                  | scale |    CelebA-HQ     |      GPU Info       |                                                                                                                      Download                                                                                                                       |
| :-------------------------------------------------------------------------------------------------------------------------------------: | :---: | :--------------: | :-----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     [dic_x8c48b6_g4_150k_CelebAHQ](https://github.com/open-mmlab/mmediting/tree/master/configs/dic/dic_x8c48b6_g4_150k_CelebAHQ.py)     |  x8   | 25.2319 / 0.7422 | 4 (Tesla PG503-216) |     [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.log.json)     |
| [dic_gan_x8c48b6_g4_500k_CelebAHQ](https://github.com/open-mmlab/mmediting/tree/master/configs/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ.py) |  x8   | 23.6241 / 0.6721 | 4 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.log.json) |

### Citation

```bibtex
@inproceedings{ma2020deep,
  title={Deep face super-resolution with iterative collaboration between attentive recovery and landmark estimation},
  author={Ma, Cheng and Jiang, Zhenyu and Rao, Yongming and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5569--5578},
  year={2020}
}
```

## DIM (CVPR'2017)

> [Deep Image Matting](https://arxiv.org/abs/1703.03872)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Image matting is a fundamental computer vision problem and has many applications. Previous algorithms have poor performance when an image has similar foreground and background colors or complicated textures. The main reasons are prior methods 1) only use low-level features and 2) lack high-level context. In this paper, we propose a novel deep learning based algorithm that can tackle both these problems. Our deep model has two parts. The first part is a deep convolutional encoder-decoder network that takes an image and the corresponding trimap as inputs and predict the alpha matte of the image. The second part is a small convolutional network that refines the alpha matte predictions of the first network to have more accurate alpha values and sharper edges. In addition, we also create a large-scale image matting dataset including 49300 training images and 1000 testing images. We evaluate our algorithm on the image matting benchmark, our testing set, and a wide variety of real images. Experimental results clearly demonstrate the superiority of our algorithm over previous methods.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175771-05b4d8f5-1abc-48ee-a5f1-8cc89a156e27.png" width="400"/>
</div >

### Results and models

|                                                         Method                                                         |   SAD    |    MSE    |   GRAD   |   CONN   | GPU Info |                                                                                                                              Download                                                                                                                               |
| :--------------------------------------------------------------------------------------------------------------------: | :------: | :-------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                     stage1 (paper)                                                     |   54.6   |   0.017   |   36.7   |   55.3   |    -     |                                                                                                                                  -                                                                                                                                  |
|                                                     stage3 (paper)                                                     | **50.4** | **0.014** |   31.0   |   50.8   |    -     |                                                                                                                                  -                                                                                                                                  |
|   [stage1 (our)](https://github.com/open-mmlab/mmediting/tree/master/configs/dim/dim_stage1_v16_1x1_1000k_comp1k.py)   |   53.8   |   0.017   |   32.7   |   54.5   |    1     |     [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage1_v16_1x1_1000k_comp1k_SAD-53.8_20200605_140257-979a420f.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage1_v16_1x1_1000k_comp1k_20200605_140257.log.json)     |
| [stage2 (our)](https://github.com/open-mmlab/mmediting/tree/master/configs/dim/dim_stage2_v16_pln_1x1_1000k_comp1k.py) |   52.3   |   0.016   |   29.4   |   52.4   |    1     | [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage2_v16_pln_1x1_1000k_comp1k_20200607_171909.log.json) |
| [stage3 (our)](https://github.com/open-mmlab/mmediting/tree/master/configs/dim/dim_stage3_v16_pln_1x1_1000k_comp1k.py) |   50.6   |   0.015   | **29.0** | **50.7** |    1     | [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_20200609_111851.log.json) |

**NOTE**

- stage1: train the encoder-decoder part without the refinement part.
- stage2: fix the encoder-decoder part and train the refinement part.
- stage3: fine-tune the whole network.

> The performance of the model is not stable during the training. Thus, the reported performance is not from the last checkpoint. Instead, it is the best performance of all validations during training.

> The performance of training (best performance) with different random seeds diverges in a large range. You may need to run several experiments for each setting to obtain the above performance.

### Citation

```bibtex
@inproceedings{xu2017deep,
  title={Deep image matting},
  author={Xu, Ning and Price, Brian and Cohen, Scott and Huang, Thomas},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2970--2979},
  year={2017}
}
```

## EDSR (CVPR'2017)

> [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Recent research on super-resolution has progressed with the development of deep convolutional neural networks (DCNN). In particular, residual learning techniques exhibit improved performance. In this paper, we develop an enhanced deep super-resolution network (EDSR) with performance exceeding those of current state-of-the-art SR methods. The significant performance improvement of our model is due to optimization by removing unnecessary modules in conventional residual networks. The performance is further improved by expanding the model size while we stabilize the training procedure. We also propose a new multi-scale deep super-resolution system (MDSR) and training method, which can reconstruct high-resolution images of different upscaling factors in a single model. The proposed methods show superior performance over the state-of-the-art methods on benchmark datasets and prove its excellence by winning the NTIRE2017 Super-Resolution Challenge.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144018090-ed629948-bf68-43ff-b2a9-6213e23f19a5.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                                              Method                                                              |       Set5       |      Set14       |      DIV2K       | GPU Info |                                                                                                                   Download                                                                                                                    |
| :------------------------------------------------------------------------------------------------------------------------------: | :--------------: | :--------------: | :--------------: | :------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [edsr_x2c64b16_1x16_300k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/edsr/edsr_x2c64b16_g1_300k_div2k.py) | 35.7592 / 0.9372 | 31.4290 / 0.8874 | 34.5896 / 0.9352 |    1     | [model](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x2c64b16_1x16_300k_div2k_20200604-19fe95ea.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x2c64b16_1x16_300k_div2k_20200604_221933.log.json) |
| [edsr_x3c64b16_1x16_300k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/edsr/edsr_x3c64b16_g1_300k_div2k.py) | 32.3301 / 0.8912 | 28.4125 / 0.8022 | 30.9154 / 0.8711 |    1     | [model](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x3c64b16_1x16_300k_div2k_20200608-36d896f4.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x3c64b16_1x16_300k_div2k_20200608_114850.log.json) |
| [edsr_x4c64b16_1x16_300k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/edsr/edsr_x4c64b16_g1_300k_div2k.py) | 30.2223 / 0.8500 | 26.7870 / 0.7366 | 28.9675 / 0.8172 |    1     | [model](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608_115148.log.json) |

### Citation

```bibtex
@inproceedings{lim2017enhanced,
  title={Enhanced deep residual networks for single image super-resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={136--144},
  year={2017}
}
```

## EDVR (CVPRW'2019)

> [EDVR: Video Restoration with Enhanced Deformable Convolutional Networks](https://arxiv.org/abs/1905.02716?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Video restoration tasks, including super-resolution, deblurring, etc, are drawing increasing attention in the computer vision community. A challenging benchmark named REDS is released in the NTIRE19 Challenge. This new benchmark challenges existing methods from two aspects: (1) how to align multiple frames given large motions, and (2) how to effectively fuse different frames with diverse motion and blur. In this work, we propose a novel Video Restoration framework with Enhanced Deformable networks, termed EDVR, to address these challenges. First, to handle large motions, we devise a Pyramid, Cascading and Deformable (PCD) alignment module, in which frame alignment is done at the feature level using deformable convolutions in a coarse-to-fine manner. Second, we propose a Temporal and Spatial Attention (TSA) fusion module, in which attention is applied both temporally and spatially, so as to emphasize important features for subsequent restoration. Thanks to these modules, our EDVR wins the champions and outperforms the second place by a large margin in all four tracks in the NTIRE19 video restoration and enhancement challenges. EDVR also demonstrates superior performance to state-of-the-art published methods on video super-resolution and deblurring.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144018263-6a1f74a4-d011-47fd-906b-290dd77eed64.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                                                           Method                                                                           |      REDS4       |         GPU Info         |                                                                                                                               Download                                                                                                                                |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: | :----------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|              [edvrm_wotsa_x4_8x4_600k_reds](https://github.com/open-mmlab/mmediting/tree/master/configs/edvr/edvrm_wotsa_x4_g8_600k_reds.py)               | 30.3430 / 0.8664 |            8             |              [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522_141644.log.json)              |
|                    [edvrm_x4_8x4_600k_reds](https://github.com/open-mmlab/mmediting/tree/master/configs/edvr/edvrm_x4_g8_600k_reds.py)                     | 30.4194 / 0.8684 |            8             |                    [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20200622_102544.log.json)                    |
| [edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4](https://github.com/open-mmlab/mmediting/tree/master/configs/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4.py) | 31.0010 / 0.8784 | 8 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4_20211228-d895a769.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_wotsa_c128b40_8x8_lr2e-4_600k_reds4_20211228_144658.log.json) |
|       [edvrl_c128b40_8x8_lr2e-4_600k_reds4](https://github.com/open-mmlab/mmediting/tree/master/configs/edvr/edvrl_c128b40_8x8_lr2e-4_600k_reds4.py)       | 31.0467 / 0.8793 | 8 (Tesla V100-PCIE-32GB) |       [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104-4509865f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104_171823.log.json)       |

### Citation

```bibtex
@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```

## ESRGAN (ECCVW'2018)

> [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144018578-6bb10830-b5fd-4d14-984e-4d7d85965c20.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                                                          Method                                                                          |       Set5        |      Set14       |      DIV2K       | GPU Info |                                                                                                                                Download                                                                                                                                 |
| :------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------: | :--------------: | :--------------: | :------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [esrgan_psnr_x4c64b23g32_1x16_1000k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py) | 30.6428 / 0.8559  | 27.0543 / 0.7447 | 29.3354 / 0.8263 |    1     | [model](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420-bf5c993c.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_psnr_x4c64b23g32_1x16_1000k_div2k_20200420_112550.log.json) |
|       [esrgan_x4c64b23g32_1x16_400k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py)       | 28.2700 /  0.7778 | 24.6328 / 0.6491 | 26.6531 / 0.7340 |    1     |       [model](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508_191042.log.json)       |

### Citation

```bibtex
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
  pages={0--0},
  year={2018}
}
```

## FLAVR (arXiv'2020)

> [FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation](https://arxiv.org/pdf/2012.08512.pdf)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Most modern frame interpolation approaches rely on explicit bidirectional optical flows between adjacent frames, thus are sensitive to the accuracy of underlying flow estimation in handling occlusions while additionally introducing computational bottlenecks unsuitable for efficient deployment. In this work, we propose a flow-free approach that is completely end-to-end trainable for multi-frame video interpolation. Our method, FLAVR, is designed to reason about non-linear motion trajectories and complex occlusions implicitly from unlabeled videos and greatly simplifies the process of training, testing and deploying frame interpolation models. Furthermore, FLAVR delivers up to 6× speed up compared to the current state-of-the-art methods for multi-frame interpolation while consistently demonstrating superior qualitative and quantitative results compared with prior methods on popular benchmarks including Vimeo-90K, Adobe-240FPS, and GoPro. Finally, we show that frame interpolation is a competitive self-supervised pre-training task for videos via demonstrating various novel applications of FLAVR including action recognition, optical flow estimation, motion magnification, and video object tracking. Code and trained models are provided in the supplementary material.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/56712176/169070212-52acdcea-d732-4441-9983-276e2e40b195.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                                                       Method                                                                        | scale | Vimeo90k-triplet  |      GPU Info       |                                                                                                                                       Download                                                                                                                                        |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: | :---: | :---------------: | :-----------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [flavr_in4out1_g8b4_vimeo90k_septuplet](https://github.com/open-mmlab/mmediting/tree/master/configs/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet.py) |  x2   | 36.3340 / 0.96015 | 8 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.log.json) |

Note: FLAVR for x8 VFI task will supported in the future.

### Citation

```bibtex
@article{kalluri2020flavr,
  title={Flavr: Flow-agnostic video representations for fast frame interpolation},
  author={Kalluri, Tarun and Pathak, Deepak and Chandraker, Manmohan and Tran, Du},
  journal={arXiv preprint arXiv:2012.08512},
  year={2020}
}
```

## GCA (AAAI'2020)

> [Natural Image Matting via Guided Contextual Attention](https://arxiv.org/abs/2001.04069)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Over the last few years, deep learning based approaches have achieved outstanding improvements in natural image matting. Many of these methods can generate visually plausible alpha estimations, but typically yield blurry structures or textures in the semitransparent area. This is due to the local ambiguity of transparent objects. One possible solution is to leverage the far-surrounding information to estimate the local opacity. Traditional affinity-based methods often suffer from the high computational complexity, which are not suitable for high resolution alpha estimation. Inspired by affinity-based method and the successes of contextual attention in inpainting, we develop a novel end-to-end approach for natural image matting with a guided contextual attention module, which is specifically designed for image matting. Guided contextual attention module directly propagates high-level opacity information globally based on the learned low-level affinity. The proposed method can mimic information flow of affinity-based methods and utilize rich features learned by deep neural networks simultaneously. Experiment results on Composition-1k testing set and this http URL benchmark dataset demonstrate that our method outperforms state-of-the-art approaches in natural image matting.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144176004-c9c26201-f8af-416a-9bea-ccd60bae7913.png" width="400"/>
</div >

### Results and models

|                                                       Method                                                       |    SAD    |    MSE     |   GRAD    |   CONN    | GPU Info |                                                                                                                         Download                                                                                                                         |
| :----------------------------------------------------------------------------------------------------------------: | :-------: | :--------: | :-------: | :-------: | :------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                  baseline (paper)                                                  |   40.62   |   0.0106   |   21.53   |   38.43   |    -     |                                                                                                                            -                                                                                                                             |
|                                                    GCA (paper)                                                     |   35.28   |   0.0091   |   16.92   |   32.53   |    -     |                                                                                                                            -                                                                                                                             |
| [baseline (our)](https://github.com/open-mmlab/mmediting/tree/master/configs/gca/baseline_r34_4x10_200k_comp1k.py) |   34.61   |   0.0083   |   16.21   |   32.12   |    4     | [model](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-34.61_20220620-96f85d56.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-34.61_20220620-96f85d56.log) |
|      [GCA (our)](https://github.com/open-mmlab/mmediting/tree/master/configs/gca/gca_r34_4x10_200k_comp1k.py)      | **33.38** | **0.0081** | **14.96** | **30.59** |    4     |      [model](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.log)      |

**More results**

|                                                                 Method                                                                  |  SAD  |  MSE   | GRAD  | CONN  | GPU Info |                                                                                                                                Download                                                                                                                                |
| :-------------------------------------------------------------------------------------------------------------------------------------: | :---: | :----: | :---: | :---: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [baseline (with DIM pipeline)](https://github.com/open-mmlab/mmediting/tree/master/configs/gca/baseline_dimaug_r34_4x10_200k_comp1k.py) | 49.95 | 0.0144 | 30.21 | 49.67 |    4     | [model](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_SAD-49.95_20200626_231612-535c9a11.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_20200626_231612.log.json) |
|      [GCA (with DIM pipeline)](https://github.com/open-mmlab/mmediting/tree/master/configs/gca/gca_dimaug_r34_4x10_200k_comp1k.py)      | 49.42 | 0.0129 | 28.07 | 49.47 |    4     |      [model](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_SAD-49.42_20200626_231422-8e9cc127.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_20200626_231422.log.json)      |

### Citation

```bibtex
@inproceedings{li2020natural,
  title={Natural Image Matting via Guided Contextual Attention},
  author={Li, Yaoyi and Lu, Hongtao},
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2020}
}
```

## GLEAN (CVPR'2021)

> [GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution](https://arxiv.org/abs/2012.00739)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We show that pre-trained Generative Adversarial Networks (GANs), e.g., StyleGAN, can be used as a latent bank to improve the restoration quality of large-factor image super-resolution (SR). While most existing SR approaches attempt to generate realistic textures through learning with adversarial loss, our method, Generative LatEnt bANk (GLEAN), goes beyond existing practices by directly leveraging rich and diverse priors encapsulated in a pre-trained GAN. But unlike prevalent GAN inversion methods that require expensive image-specific optimization at runtime, our approach only needs a single forward pass to generate the upscaled image. GLEAN can be easily incorporated in a simple encoder-bank-decoder architecture with multi-resolution skip connections. Switching the bank allows the method to deal with images from diverse categories, e.g., cat, building, human face, and car. Images upscaled by GLEAN show clear improvements in terms of fidelity and texture faithfulness in comparison to existing methods.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144019196-2642f3be-f82e-4fa4-8d96-4161354db9a7.png" width="400"/>
</div >

### Results and models

For the meta info used in training and test, please refer to [here](https://github.com/ckkelvinchan/GLEAN). The results are evaluated on RGB channels.

|                                                                           Method                                                                            | PSNR  |         GPU Info         |                                                                                                                                Download                                                                                                                                 |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------: | :---: | :----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                              [glean_cat_8x](https://github.com/open-mmlab/mmediting/tree/master/configs/glean/glean_cat_8x.py)                              | 23.98 | 2 (Tesla V100-PCIE-32GB) |                              [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614-d3ac8683.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614_145540.log.json)                              |
|                            [glean_ffhq_16x](https://github.com/open-mmlab/mmediting/tree/master/configs/glean/glean_ffhq_16x.py)                            | 26.91 | 2 (Tesla V100-PCIE-32GB) |                            [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527-61a3afad.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527_194536.log.json)                            |
|                             [glean_cat_16x](https://github.com/open-mmlab/mmediting/tree/master/configs/glean/glean_cat_16x.py)                             | 20.88 | 2 (Tesla V100-PCIE-32GB) |                             [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527-68912543.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527_103708.log.json)                             |
| [glean_in128out1024_4x2_300k_ffhq_celebahq](https://github.com/open-mmlab/mmediting/tree/master/configs/glean/glean_in128out1024_4x2_300k_ffhq_celebahq.py) | 27.94 | 4 (Tesla V100-SXM3-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812_100549.log.json) |

### Citation

```bibtex
@InProceedings{chan2021glean,
  author = {Chan, Kelvin CK and Wang, Xintao and Xu, Xiangyu and Gu, Jinwei and Loy, Chen Change},
  title = {GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

## Global&Local (ToG'2017)

> [Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We present a novel approach for image completion that results in images that are both locally and globally consistent. With a fully-convolutional neural network, we can complete images of arbitrary resolutions by flling in missing regions of any shape. To train this image completion network to be consistent, we use global and local context discriminators that are trained to distinguish real images from completed ones. The global discriminator looks at the entire image to assess if it is coherent as a whole, while the local discriminator looks only at a small area centered at the completed region to ensure the local consistency of the generated patches. The image completion network is then trained to fool the both context discriminator networks, which requires it to generate images that are indistinguishable from real ones with regard to overall consistency as well as in details. We show that our approach can be used to complete a wide variety of scenes. Furthermore, in contrast with the patch-based approaches such as PatchMatch, our approach can generate fragments that do not appear elsewhere in the image, which allows us to naturally complete the image.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175196-51dfda11-f7e1-4c7e-abed-42799f757bef.png" width="400"/>
</div >

### Results and models

*Note that we do not apply the post-processing module in Global&Local for a fair comparison with current deep inpainting methods.*

**Places365-Challenge**

|                                                       Method                                                       |  Mask Type  | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  | GPU Info |                                                                                                                      Download                                                                                                                       |
| :----------------------------------------------------------------------------------------------------------------: | :---------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Global&Local](https://github.com/open-mmlab/mmediting/tree/master/configs/global_local/gl_256x256_8x12_places.py) | square bbox |  256x256   |    500k     | Places365-val |  11.164  | 23.152 | 0.862 |    8     | [model](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.log.json) |

**CelebA-HQ**

|                                                       Method                                                       |  Mask Type  | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  | GPU Info |                                                                                                                      Download                                                                                                                       |
| :----------------------------------------------------------------------------------------------------------------: | :---------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Global&Local](https://github.com/open-mmlab/mmediting/tree/master/configs/global_local/gl_256x256_8x12_celeba.py) | square bbox |  256x256   |    500k     | CelebA-val |  6.678   | 26.780 | 0.904 |    8     | [model](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.log.json) |

### Citation

```bibtex
@article{iizuka2017globally,
  title={Globally and locally consistent image completion},
  author={Iizuka, Satoshi and Simo-Serra, Edgar and Ishikawa, Hiroshi},
  journal={ACM Transactions on Graphics (ToG)},
  volume={36},
  number={4},
  pages={1--14},
  year={2017},
  publisher={ACM New York, NY, USA}
}
```

## IconVSR (CVPR'2021)

> [BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond](https://arxiv.org/abs/2012.02181)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Video super-resolution (VSR) approaches tend to have more components than the image counterparts as they need to exploit the additional temporal dimension. Complex designs are not uncommon. In this study, we wish to untangle the knots and reconsider some most essential components for VSR guided by four basic functionalities, i.e., Propagation, Alignment, Aggregation, and Upsampling. By reusing some existing components added with minimal redesigns, we show a succinct pipeline, BasicVSR, that achieves appealing improvements in terms of speed and restoration quality in comparison to many state-of-the-art algorithms. We conduct systematic analysis to explain how such gain can be obtained and discuss the pitfalls. We further show the extensibility of BasicVSR by presenting an information-refill mechanism and a coupled propagation scheme to facilitate information aggregation. The BasicVSR and its extension, IconVSR, can serve as strong baselines for future VSR approaches.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144011348-c58101d4-5f69-4e58-be2b-7accd07b06fa.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels for REDS4 and Y channel for others. The metrics are `PSNR` / `SSIM` .
The pretrained weights of the IconVSR components can be found here: [SPyNet](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth), [EDVR-M for REDS](https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_reds_20210413-3867262f.pth), and [EDVR-M for Vimeo-90K](https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_vimeo90k_20210413-e40e99a8.pth).

|                                                      Method                                                       | REDS4 (BIx4)<br>PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4)<br>PSNR/SSIM (Y) | Vid4 (BIx4)<br>PSNR/SSIM (Y) | UDM10 (BDx4)<br>PSNR/SSIM (Y) | Vimeo-90K-T (BDx4)<br>PSNR/SSIM (Y) | Vid4 (BDx4)<br>PSNR/SSIM (Y) |         GPU Info         |                                                                                                            Download                                                                                                             |
| :---------------------------------------------------------------------------------------------------------------: | :-----------------------------: | :---------------------------------: | :--------------------------: | :---------------------------: | :---------------------------------: | :--------------------------: | :----------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [iconvsr_reds4](https://github.com/open-mmlab/mmediting/tree/master/configs/iconvsr/iconvsr_reds4.py)       |       **31.6926/0.8951**        |           36.4983/0.9416            |      **27.4809/0.8354**      |        35.3377/0.9471         |           34.4299/0.9287            |        25.2110/0.7732        | 2 (Tesla V100-PCIE-32GB) |       [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413_222735.log.json)       |
| [iconvsr_vimeo90k_bi](https://github.com/open-mmlab/mmediting/tree/master/configs/iconvsr/iconvsr_vimeo90k_bi.py) |         30.3452/0.8659          |         **37.3729/0.9467**          |        27.4238/0.8297        |        34.2595/0.9398         |           34.5548/0.9295            |        24.6666/0.7491        | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413-7c7418dc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413_222757.log.json) |
| [iconvsr_vimeo90k_bd](https://github.com/open-mmlab/mmediting/tree/master/configs/iconvsr/iconvsr_vimeo90k_bd.py) |         29.0150/0.8465          |           34.6780/0.9339            |        26.3109/0.8028        |      **40.0640/0.9697**       |         **37.7573/0.9517**          |      **28.2464/0.8612**      | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414-5f38cb34.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414_084128.log.json) |

### Citation

```bibtex
@InProceedings{chan2021basicvsr,
  author = {Chan, Kelvin CK and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title = {BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

## IndexNet (ICCV'2019)

> [Indices Matter: Learning to Index for Deep Image Matting](https://arxiv.org/abs/1908.00672)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We show that existing upsampling operators can be unified with the notion of the index function. This notion is inspired by an observation in the decoding process of deep image matting where indices-guided unpooling can recover boundary details much better than other upsampling operators such as bilinear interpolation. By looking at the indices as a function of the feature map, we introduce the concept of learning to index, and present a novel index-guided encoder-decoder framework where indices are self-learned adaptively from data and are used to guide the pooling and upsampling operators, without the need of supervision. At the core of this framework is a flexible network module, termed IndexNet, which dynamically predicts indices given an input. Due to its flexibility, IndexNet can be used as a plug-in applying to any off-the-shelf convolutional networks that have coupled downsampling and upsampling stages.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144176083-52604501-1f46-411d-b81a-cad0eb4b529f.png" width="400"/>
</div >

### Results and models

|                                                          Method                                                          |   SAD    |    MSE    |   GRAD   |   CONN   | GPU Info |                                                                                                                              Download                                                                                                                               |
| :----------------------------------------------------------------------------------------------------------------------: | :------: | :-------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                     M2O DINs (paper)                                                     |   45.8   |   0.013   |   25.9   | **43.7** |    -     |                                                                                                                                  -                                                                                                                                  |
| [M2O DINs (our)](https://github.com/open-mmlab/mmediting/tree/master/configs/indexnet/indexnet_mobv2_1x16_78k_comp1k.py) | **45.6** | **0.012** | **25.5** |   44.8   |    1     | [model](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_20200618_173817.log.json) |

> The performance of training (best performance) with different random seeds diverges in a large range. You may need to run several experiments for each setting to obtain the above performance.

**More result**

|                                                                    Method                                                                     | SAD  |  MSE  | GRAD | CONN | GPU Info |                                                                                                                                     Download                                                                                                                                      |
| :-------------------------------------------------------------------------------------------------------------------------------------------: | :--: | :---: | :--: | :--: | :------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [M2O DINs (with DIM pipeline)](https://github.com/open-mmlab/mmediting/tree/master/configs/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k.py) | 50.1 | 0.016 | 30.8 | 49.5 |    1     | [model](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k_SAD-50.1_20200626_231857-af359436.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k_20200626_231857.log.json) |

### Citation

```bibtex
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## LIIF (CVPR'2021)

> [Learning Continuous Image Representation with Local Implicit Image Function](https://arxiv.org/abs/2012.09161)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

How to represent an image? While the visual world is presented in a continuous manner, machines store and see the images in a discrete way with 2D arrays of pixels. In this paper, we seek to learn a continuous representation for images. Inspired by the recent progress in 3D reconstruction with implicit neural representation, we propose Local Implicit Image Function (LIIF), which takes an image coordinate and the 2D deep features around the coordinate as inputs, predicts the RGB value at a given coordinate as an output. Since the coordinates are continuous, LIIF can be presented in arbitrary resolution. To generate the continuous representation for images, we train an encoder with LIIF representation via a self-supervised task with super-resolution. The learned continuous representation can be presented in arbitrary resolution even extrapolate to x30 higher resolution, where the training tasks are not provided. We further show that LIIF representation builds a bridge between discrete and continuous representation in 2D, it naturally supports the learning tasks with size-varied image ground-truths and significantly outperforms the method with resizing the ground-truths.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144032669-da59d683-9c4f-4598-a680-32770a369b74.png" width="400"/>
</div >

### Results and models

|                                                                      Method                                                                      | scale | Set5<br>PSNR / SSIM | Set14<br>PSNR / SSIM | DIV2K <br>PSNR / SSIM |   GPU Info   |                                                                                                                           Download                                                                                                                            |
| :----------------------------------------------------------------------------------------------------------------------------------------------: | :---: | :-----------------: | :------------------: | :-------------------: | :----------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [liif_edsr_norm_c64b16_g1_1000k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/liif/liif_edsr_norm_c64b16_g1_1000k_div2k.py) |  x2   |  35.7131 / 0.9366   |   31.5579 / 0.8889   |   34.6647 / 0.9355    | 1 (TITAN Xp) | [model](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.log.json) |
|                                                                        △                                                                         |  x3   |  32.3805 / 0.8915   |   28.4605 / 0.8039   |   30.9808 / 0.8724    |      △       |                                                                                                                               △                                                                                                                               |
|                                                                        △                                                                         |  x4   |  30.2748 / 0.8509   |   26.8415 / 0.7381   |   29.0245 / 0.8187    |      △       |                                                                                                                               △                                                                                                                               |
|                                                                        △                                                                         |  x6   |  27.1187 / 0.7774   |   24.7461 / 0.6444   |   26.7770 / 0.7425    |      △       |                                                                                                                               △                                                                                                                               |
|                                                                        △                                                                         |  x18  |  20.8516 / 0.5406   |   20.0096 / 0.4525   |   22.1987 / 0.5955    |      △       |                                                                                                                               △                                                                                                                               |
|                                                                        △                                                                         |  x30  |  18.8467 / 0.5010   |   18.1321 / 0.3963   |   20.5050 / 0.5577    |      △       |                                                                                                                               △                                                                                                                               |
|  [liif_rdn_norm_c64b16_g1_1000k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/liif/liif_rdn_norm_c64b16_g1_1000k_div2k.py)  |  x2   |  35.7874 / 0.9366   |   31.6866 / 0.8896   |   34.7548 / 0.9356    | 1 (TITAN Xp) |  [model](https://download.openmmlab.com/mmediting/restorers/liif/liif_rdn_norm_c64b16_g1_1000k_div2k_20210717-22d6fdc8.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/liif/liif_rdn_norm_c64b16_g1_1000k_div2k_20210717-22d6fdc8.log.json)  |
|                                                                        △                                                                         |  x3   |  32.4992 / 0.8923   |   28.4905 / 0.8037   |   31.0744 / 0.8731    |      △       |                                                                                                                               △                                                                                                                               |
|                                                                        △                                                                         |  x4   |  30.3835 / 0.8513   |   26.8734 / 0.7373   |   29.1101 / 0.8197    |      △       |                                                                                                                               △                                                                                                                               |
|                                                                        △                                                                         |  x6   |  27.1914 / 0.7751   |   24.7824 / 0.6434   |   26.8693 / 0.7437    |      △       |                                                                                                                               △                                                                                                                               |
|                                                                        △                                                                         |  x18  |  20.8913 / 0.5329   |   20.1077 / 0.4537   |   22.2972 / 0.5950    |      △       |                                                                                                                               △                                                                                                                               |
|                                                                        △                                                                         |  x30  |  18.9354 / 0.4864   |   18.1448 / 0.3942   |   20.5663 / 0.5560    |      △       |                                                                                                                               △                                                                                                                               |

Note:

- △ refers to ditto.
- Evaluated on RGB channels,  `scale` pixels in each border are cropped before evaluation.

### Citation

```bibtex
@inproceedings{chen2021learning,
  title={Learning continuous image representation with local implicit image function},
  author={Chen, Yinbo and Liu, Sifei and Wang, Xiaolong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8628--8638},
  year={2021}
}
```

## PConv (ECCV'2018)

> [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Existing deep learning based image inpainting methods use a standard convolutional network over the corrupted image, using convolutional filter responses conditioned on both valid pixels as well as the substitute values in the masked holes (typically the mean value). This often leads to artifacts such as color discrepancy and blurriness. Post-processing is usually used to reduce such artifacts, but are expensive and may fail. We propose the use of partial convolutions, where the convolution is masked and renormalized to be conditioned on only valid pixels. We further include a mechanism to automatically generate an updated mask for the next layer as part of the forward pass. Our model outperforms other methods for irregular masks. We show qualitative and quantitative comparisons with other methods to validate our approach.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175613-1bc9ad1b-072d-4c1f-a97d-1af5be2590bd.png" width="400"/>
</div >

### Results and models

**Places365-Challenge**

|                                                        Method                                                        | Mask Type | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  | GPU Info |                                                                                                                        Download                                                                                                                         |
| :------------------------------------------------------------------------------------------------------------------: | :-------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PConv](https://github.com/open-mmlab/mmediting/tree/master/configs/partial_conv/pconv_256x256_stage2_4x2_places.py) | free-form |  256x256   |    500k     | Places365-val |  8.776   | 22.762 | 0.801 |    4     | [model](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.log.json) |

**CelebA-HQ**

|                                                        Method                                                        | Mask Type | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  | GPU Info |                                                                                                                        Download                                                                                                                         |
| :------------------------------------------------------------------------------------------------------------------: | :-------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PConv](https://github.com/open-mmlab/mmediting/tree/master/configs/partial_conv/pconv_256x256_stage2_4x2_celeba.py) | free-form |  256x256   |    500k     | CelebA-val |  5.990   | 25.404 | 0.853 |    4     | [model](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.log.json) |

### Citation

```bibtex
@inproceedings{liu2018image,
  title={Image inpainting for irregular holes using partial convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={85--100},
  year={2018}
}
```

## RDN (CVPR'2018)

> [Residual Dense Network for Image Super-Resolution](https://arxiv.org/abs/1802.08797)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

A very deep convolutional neural network (CNN) has recently achieved great success for image super-resolution (SR) and offered hierarchical features as well. However, most deep CNN based SR models do not make full use of the hierarchical features from the original low-resolution (LR) images, thereby achieving relatively-low performance. In this paper, we propose a novel residual dense network (RDN) to address this problem in image SR. We fully exploit the hierarchical features from all the convolutional layers. Specifically, we propose residual dense block (RDB) to extract abundant local features via dense connected convolutional layers. RDB further allows direct connections from the state of preceding RDB to all the layers of current RDB, leading to a contiguous memory (CM) mechanism. Local feature fusion in RDB is then used to adaptively learn more effective features from preceding and current local features and stabilizes the training of wider network. After fully obtaining dense local features, we use global feature fusion to jointly and adaptively learn global hierarchical features in a holistic way. Extensive experiments on benchmark datasets with different degradation models show that our RDN achieves favorable performance against state-of-the-art methods.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144034203-c3a4ac55-d815-4180-a345-f80ab5ca68b6.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                                            Method                                                             |       Set5       |      Set14       |      DIV2K       |   GPU Info   |                                                                                                                 Download                                                                                                                  |
| :---------------------------------------------------------------------------------------------------------------------------: | :--------------: | :--------------: | :--------------: | :----------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [rdn_x2c64b16_g1_1000k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/rdn/rdn_x2c64b16_g1_1000k_div2k.py) | 35.9883 / 0.9385 | 31.8366 / 0.8920 | 34.9392 / 0.9380 | 1 (TITAN Xp) | [model](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_20210419-dc146009.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_20210419-dc146009.log.json) |
| [rdn_x3c64b16_g1_1000k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/rdn/rdn_x3c64b16_g1_1000k_div2k.py) | 32.6051 / 0.8943 | 28.6338 / 0.8077 | 31.2153 / 0.8763 | 1 (TITAN Xp) | [model](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x3c64b16_g1_1000k_div2k_20210419-b93cb6aa.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x3c64b16_g1_1000k_div2k_20210419-b93cb6aa.log.json) |
| [rdn_x4c64b16_g1_1000k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/rdn/rdn_x4c64b16_g1_1000k_div2k.py) | 30.4922 / 0.8548 | 26.9570 / 0.7423 | 29.1925 / 0.8233 | 1 (TITAN Xp) | [model](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.log.json) |

### Citation

```bibtex
@inproceedings{zhang2018residual,
  title={Residual dense network for image super-resolution},
  author={Zhang, Yulun and Tian, Yapeng and Kong, Yu and Zhong, Bineng and Fu, Yun},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2472--2481},
  year={2018}
}
```

## RealBasicVSR (CVPR'2022)

> [RealBasicVSR: Investigating Tradeoffs in Real-World Video Super-Resolution](https://arxiv.org/abs/2111.12704)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

The diversity and complexity of degradations in real-world video super-resolution (VSR) pose non-trivial challenges in inference and training. First, while long-term propagation leads to improved performance in cases of mild degradations, severe in-the-wild degradations could be exaggerated through propagation, impairing output quality. To balance the tradeoff between detail synthesis and artifact suppression, we found an image pre-cleaning stage indispensable to reduce noises and artifacts prior to propagation. Equipped with a carefully designed cleaning module, our RealBasicVSR outperforms existing methods in both quality and efficiency. Second, real-world VSR models are often trained with diverse degradations to improve generalizability, requiring increased batch size to produce a stable gradient. Inevitably, the increased computational burden results in various problems, including 1) speed-performance tradeoff and 2) batch-length tradeoff. To alleviate the first tradeoff, we propose a stochastic degradation scheme that reduces up to 40% of training time without sacrificing performance. We then analyze different training settings and suggest that employing longer sequences rather than larger batches during training allows more effective uses of temporal information, leading to more stable performance during inference. To facilitate fair comparisons, we propose the new VideoLQ dataset, which contains a large variety of real-world low-quality video sequences containing rich textures and patterns. Our dataset can serve as a common ground for benchmarking. Code, models, and the dataset will be made publicly available.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/146704029-58bc4db4-267f-4158-8129-e49ab6652249.png" width="400"/>
</div >

### Results and models

Evaluated on Y channel. The code for computing NRQM, NIQE, and PI can be found [here](https://github.com/roimehrez/PIRM2018). MATLAB official code is used to compute BRISQUE.

|                                                                                 Method                                                                                  | NRQM (Y) | NIQE (Y) | PI (Y) | BRISQUE (Y) |         GPU Info         |                                                                                                                                         Download                                                                                                                                         |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------: | :------: | :----: | :---------: | :----------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds](https://github.com/open-mmlab/mmediting/tree/master/configs/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py) |  6.0477  |  3.7662  | 3.8593 |   29.030    | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth)/[log](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104_183640.log.json) |

### Citation

```bibtex
@InProceedings{chan2022investigating,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {RealBasicVSR: Investigating Tradeoffs in Real-World Video Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2022}
}
```

## Real-ESRGAN (ICCVW'2021)

> [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Though many attempts have been made in blind super-resolution to restore low-resolution images with unknown and complex degradations, they are still far from addressing general real-world degraded images. In this work, we extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data. Specifically, a high-order degradation modeling process is introduced to better simulate complex real-world degradations. We also consider the common ringing and overshoot artifacts in the synthesis process. In addition, we employ a U-Net discriminator with spectral normalization to increase discriminator capability and stabilize the training dynamics. Extensive comparisons have shown its superior visual performance than prior works on various real datasets. We also provide efficient implementations to synthesize training pairs on the fly.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144034533-f81430df-351b-490c-9e00-733465edf3ee.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels. The metrics are `PSNR/SSIM`.

|                                                                                    Method                                                                                     |      Set5      |         GPU Info         |                                                                                                                                          Download                                                                                                                                           |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------: | :----------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost](https://github.com/open-mmlab/mmediting/tree/master/configs/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost.py) | 28.0297/0.8236 | 4 (Tesla V100-SXM2-32GB) |                                                                      [model](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost_20210816-4ae3b5a4.pth)/log                                                                      |
|  [realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost](https://github.com/open-mmlab/mmediting/tree/master/configs/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost.py)  | 26.2204/0.7655 | 4 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20211010-34798885.pth) /[log](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20210922_142838.log.json) |

### Citation

```bibtex
@inproceedings{wang2021real,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={1905--1914},
  year={2021}
}
```

## SRCNN (TPAMI'2015)

> [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN) that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional sparse-coding-based SR methods can also be viewed as a deep convolutional network. But unlike traditional methods that handle each component separately, our method jointly optimizes all layers. Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. We explore different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144034831-79f48aae-196e-42e7-92b9-069149733e3e.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                                              Method                                                               |       Set5       |       Set14       |      DIV2K       | GPU Info |                                                                                                                    Download                                                                                                                     |
| :-------------------------------------------------------------------------------------------------------------------------------: | :--------------: | :---------------: | :--------------: | :------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [srcnn_x4k915_1x16_1000k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/srcnn/srcnn_x4k915_g1_1000k_div2k.py) | 28.4316 / 0.8099 | 25.6486 /  0.7014 | 27.7460 / 0.7854 |    1     | [model](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608_120159.log.json) |

### Citation

```bibtex
@article{dong2015image,
  title={Image super-resolution using deep convolutional networks},
  author={Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={38},
  number={2},
  pages={295--307},
  year={2015},
  publisher={IEEE}
}
```

## SRGAN (CVPR'2016)

> [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035016-8ed4a80b-2d8b-4947-848b-3f8e917a9273.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.

The metrics are `PSNR / SSIM` .

|                                                                       Method                                                                        |       Set5        |      Set14       |      DIV2K       | GPU Info |                                                                                                                                  Download                                                                                                                                   |
| :-------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------: | :--------------: | :--------------: | :------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [msrresnet_x4c64b16_1x16_300k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/srgan_resnet/msrresnet_x4c64b16_g1_1000k_div2k.py) | 30.2252 / 0.8491  | 26.7762 / 0.7369 | 28.9748 / 0.8178 |    1     | [model](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521-61556be5.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521_110246.log.json) |
|    [srgan_x4c64b16_1x16_1000k_div2k](https://github.com/open-mmlab/mmediting/tree/master/configs/srgan_resnet/srgan_x4c64b16_g1_1000k_div2k.py)     | 27.9499 /  0.7846 | 24.7383 / 0.6491 | 26.5697 / 0.7365 |    1     |    [model](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200606-a1f0810e.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200506_191442.log.json)    |

### Citation

```bibtex
@inproceedings{ledig2016photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  year={2016}
}
```

## TDAN (CVPR'2020)

> [TDAN: Temporally Deformable Alignment Network for Video Super-Resolution](https://arxiv.org/abs/1812.02898)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Video super-resolution (VSR) aims to restore a photo-realistic high-resolution (HR) video frame from both its corresponding low-resolution (LR) frame (reference frame) and multiple neighboring frames (supporting frames). Due to varying motion of cameras or objects, the reference frame and each support frame are not aligned. Therefore, temporal alignment is a challenging yet important problem for VSR. Previous VSR methods usually utilize optical flow between the reference frame and each supporting frame to wrap the supporting frame for temporal alignment. Therefore, the performance of these image-level wrapping-based models will highly depend on the prediction accuracy of optical flow, and inaccurate optical flow will lead to artifacts in the wrapped supporting frames, which also will be propagated into the reconstructed HR video frame. To overcome the limitation, in this paper, we propose a temporal deformable alignment network (TDAN) to adaptively align the reference frame and each supporting frame at the feature level without computing optical flow. The TDAN uses features from both the reference frame and each supporting frame to dynamically predict offsets of sampling convolution kernels. By using the corresponding kernels, TDAN transforms supporting frames to align with the reference frame. To predict the HR video frame, a reconstruction network taking aligned frames and the reference frame is utilized. Experimental results demonstrate the effectiveness of the proposed TDAN-based VSR model.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035224-a87cc41e-1352-4ffa-8b07-eda5ace8a0b1.png" width="400"/>
</div >

### Results and models

Evaluated on Y-channel. 8 pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                                                   Method                                                                   |   Vid4 (BIx4)   | SPMCS-30 (BIx4) |   Vid4 (BDx4)   | SPMCS-30 (BDx4) |         GPU Info         |                                                                                                        Download                                                                                                         |
| :----------------------------------------------------------------------------------------------------------------------------------------: | :-------------: | :-------------: | :-------------: | :-------------: | :----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [tdan_vimeo90k_bix4_ft_lr5e-5_400k](https://github.com/open-mmlab/mmediting/tree/master/configs/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py) | **26.49/0.792** | **30.42/0.856** |   25.93/0.772   |   29.69/0.842   | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_vimeo90k_bdx4_ft_lr5e-5_800k](https://github.com/open-mmlab/mmediting/tree/master/configs/tdan/tdan_vimeo90k_bdx4_ft_lr5e-5_800k.py) |   25.80/0.784   |   29.56/0.851   | **26.87/0.815** | **30.77/0.868** | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following command to train a model.

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

TDAN is trained with two stages.

**Stage 1**: Train with a larger learning rate (1e-4)

```shell
./tools/dist_train.sh configs/tdan/tdan_vimeo90k_bix4_lr1e-4_400k.py 8
```

**Stage 2**: Fine-tune with a smaller learning rate (5e-5)

```shell
./tools/dist_train.sh configs/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py 8
```

For more details, you can refer to **Train a model** part in [getting_started](en/getting_started.md##train-a-model).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--save-path ${IMAGE_SAVE_PATH}]
```

Example: Test TDAN on SPMCS-30 using Bicubic downsampling.

```shell
python tools/test.py configs/tdan/tdan_vimeo90k_bix4_ft_lr5e-5_400k.py  checkpoints/SOME_CHECKPOINT.pth --save_path outputs/
```

For more details, you can refer to **Inference with pretrained models** part in [getting_started](en/getting_started.md##inference-with-pretrained-models).

</details>

### Citation

```bibtex
@InProceedings{tian2020tdan,
  title={TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution},
  author={Tian, Yapeng and Zhang, Yulun and Fu, Yun and Xu, Chenliang},
  booktitle = {Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```

## TOFlow (IJCV'2019)

> [Video Enhancement with Task-Oriented Flow](https://arxiv.org/abs/1711.09078)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Many video enhancement algorithms rely on optical flow to register frames in a video sequence. Precise flow estimation is however intractable; and optical flow itself is often a sub-optimal representation for particular video processing tasks. In this paper, we propose task-oriented flow (TOFlow), a motion representation learned in a self-supervised, task-specific manner. We design a neural network with a trainable motion estimation component and a video processing component, and train them jointly to learn the task-oriented flow. For evaluation, we build Vimeo-90K, a large-scale, high-quality video dataset for low-level video processing. TOFlow outperforms traditional optical flow on standard benchmarks as well as our Vimeo-90K dataset in three video processing tasks: frame interpolation, video denoising/deblocking, and video super-resolution.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035477-2480d580-1409-4a7c-88d5-c13a3dbd62ac.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                                                               Method                                                                                |                                                                Pretrained SPyNet                                                                | Vimeo90k-triplet |      GPU Info       |                                                                                                                                                 Download                                                                                                                                                  |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: | :-----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        [tof_vfi_spynet_chair_nobn_1xb1_vimeo90k](https://github.com/open-mmlab/mmediting/tree/master/configs/tof/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k.py)        |    [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth)     | 33.3294 / 0.9465 | 1 (Tesla PG503-216) |        [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.log.json)        |
|        [tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k](https://github.com/open-mmlab/mmediting/tree/master/configs/tof/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k.py)        |    [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_kitti_20220321-dbcc1cc1.pth)     | 33.3339 / 0.9466 | 1 (Tesla PG503-216) |        [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.log.json)        |
| [tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k](https://github.com/open-mmlab/mmediting/tree/master/configs/tof/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_clean_20220321-0756630b.pth) | 33.3170 / 0.9464 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.log.json) |
| [tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k](https://github.com/open-mmlab/mmediting/tree/master/configs/tof/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_final_20220321-5e89dcec.pth) | 33.3237 / 0.9465 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.log.json) |
|     [tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k](https://github.com/open-mmlab/mmediting/tree/master/configs/tof/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k.py)     |   [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_pytoflow_20220321-5bab842d.pth)   | 33.3426 / 0.9467 | 1 (Tesla PG503-216) |     [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.log.json)     |

Note: These pretrained SPyNets don't contain BN layer since `batch_size=1`, which is consistent with `https://github.com/Coldog2333/pytoflow`.

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                                         Method                                                          |       Vid4       | GPU Info |                                               Download                                                |
| :---------------------------------------------------------------------------------------------------------------------: | :--------------: | :------: | :---------------------------------------------------------------------------------------------------: |
| [tof_x4_vimeo90k_official](https://github.com/open-mmlab/mmediting/tree/master/configs/tof/tof_x4_vimeo90k_official.py) | 24.4377 / 0.7433 |    -     | [model](https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth) |

### Citation

```bibtex
@article{xue2019video,
  title={Video enhancement with task-oriented flow},
  author={Xue, Tianfan and Chen, Baian and Wu, Jiajun and Wei, Donglai and Freeman, William T},
  journal={International Journal of Computer Vision},
  volume={127},
  number={8},
  pages={1106--1125},
  year={2019},
  publisher={Springer}
}
```

## TTSR (CVPR'2020)

> [Learning Texture Transformer Network for Image Super-Resolution](https://arxiv.org/abs/2006.04139)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We study on image super-resolution (SR), which aims to recover realistic textures from a low-resolution (LR) image. Recent progress has been made by taking high-resolution images as references (Ref), so that relevant textures can be transferred to LR images. However, existing SR approaches neglect to use attention mechanisms to transfer high-resolution (HR) textures from Ref images, which limits these approaches in challenging cases. In this paper, we propose a novel Texture Transformer Network for Image Super-Resolution (TTSR), in which the LR and Ref images are formulated as queries and keys in a transformer, respectively. TTSR consists of four closely-related modules optimized for image generation tasks, including a learnable texture extractor by DNN, a relevance embedding module, a hard-attention module for texture transfer, and a soft-attention module for texture synthesis. Such a design encourages joint feature learning across LR and Ref images, in which deep feature correspondences can be discovered by attention, and thus accurate texture features can be transferred. The proposed texture transformer can be further stacked in a cross-scale way, which enables texture recovery from different levels (e.g., from 1x to 4x magnification). Extensive experiments show that TTSR achieves significant improvements over state-of-the-art approaches on both quantitative and qualitative evaluations.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035689-e5afa799-f469-40a0-aa94-0b84a46726a1.png" width="400"/>
</div >

### Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                                                  Method                                                                  | scale |      CUFED       |   GPU Info   |                                                                                                                       Download                                                                                                                        |
| :--------------------------------------------------------------------------------------------------------------------------------------: | :---: | :--------------: | :----------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [ttsr-rec_x4_c64b16_g1_200k_CUFED](https://github.com/open-mmlab/mmediting/tree/master/configs/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED.py) |  x4   | 25.2433 / 0.7491 | 1 (TITAN Xp) | [model](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.log.json) |
| [ttsr-gan_x4_c64b16_g1_500k_CUFED](https://github.com/open-mmlab/mmediting/tree/master/configs/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED.py) |  x4   | 24.6075 / 0.7234 | 1 (TITAN Xp) | [model](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.log.json) |

### Citation

```bibtex
@inproceedings{yang2020learning,
  title={Learning texture transformer network for image super-resolution},
  author={Yang, Fuzhi and Yang, Huan and Fu, Jianlong and Lu, Hongtao and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5791--5800},
  year={2020}
}
```
