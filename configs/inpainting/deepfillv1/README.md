# DeepFillv1 (CVPR'2018)

> [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Recent deep learning based approaches have shown promising results for the challenging task of inpainting large missing regions in an image. These methods can generate visually plausible image structures and textures, but often create distorted structures or blurry textures inconsistent with surrounding areas. This is mainly due to ineffectiveness of convolutional neural networks in explicitly borrowing or copying information from distant spatial locations. On the other hand, traditional texture and patch synthesis approaches are particularly suitable when it needs to borrow textures from the surrounding regions. Motivated by these observations, we propose a new deep generative model-based approach which can not only synthesize novel image structures but also explicitly utilize surrounding image features as references during network training to make better predictions. The model is a feed-forward, fully convolutional neural network which can process images with multiple holes at arbitrary locations and with variable sizes during the test time. Experiments on multiple datasets including faces (CelebA, CelebA-HQ), textures (DTD) and natural images (ImageNet, Places2) demonstrate that our proposed approach generates higher-quality inpainting results than existing ones.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144174665-9675931f-e448-4475-a659-99b65e7d4a64.png" width="400"/>
</div >

## Results and models

**Places365-Challenge**

|                               Method                                |  Mask Type  | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  |                               Download                                |
| :-----------------------------------------------------------------: | :---------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :-------------------------------------------------------------------: |
| [DeepFillv1](/configs/inpainting/deepfillv1/deepfillv1_256x256_8x2_places.py) | square bbox |  256x256   |    3500k    | Places365-val |  11.019  | 23.429 | 0.862 | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.log.json) |

**CelebA-HQ**

|                                Method                                |  Mask Type  | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  |                                Download                                 |
| :------------------------------------------------------------------: | :---------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :---------------------------------------------------------------------: |
| [DeepFillv1](/configs/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba.py) | square bbox |  256x256   |    1500k    | CelebA-val |  6.677   | 26.878 | 0.911 | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.log.json) |

## Citation

```bibtex
@inproceedings{yu2018generative,
  title={Generative image inpainting with contextual attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5505--5514},
  year={2018}
}
```
