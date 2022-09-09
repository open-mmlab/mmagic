# DeepFillv2 (CVPR'2019)

> [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present a generative image inpainting system to complete images with free-form mask and guidance. The system is based on gated convolutions learned from millions of images without additional labelling efforts. The proposed gated convolution solves the issue of vanilla convolution that treats all input pixels as valid ones, generalizes partial convolution by providing a learnable dynamic feature selection mechanism for each channel at each spatial location across all layers. Moreover, as free-form masks may appear anywhere in images with any shape, global and local GANs designed for a single rectangular mask are not applicable. Thus, we also present a patch-based GAN loss, named SN-PatchGAN, by applying spectral-normalized discriminator on dense image patches. SN-PatchGAN is simple in formulation, fast and stable in training. Results on automatic image inpainting and user-guided extension demonstrate that our system generates higher-quality and more flexible results than previous methods. Our system helps user quickly remove distracting objects, modify image layouts, clear watermarks and edit faces.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175160-75473789-924f-490b-ab25-4c4f252fa55f.png" width="400"/>
</div >

## Results and models

**Places365-Challenge**

|                                Method                                | Mask Type | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  |                                Download                                |
| :------------------------------------------------------------------: | :-------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :--------------------------------------------------------------------: |
| [DeepFillv2](/configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_places.py) | free-form |  256x256   |    100k     | Places365-val |  8.635   | 22.398 | 0.815 | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.log.json) |

**CelebA-HQ**

|                                Method                                 | Mask Type | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  |                                 Download                                 |
| :-------------------------------------------------------------------: | :-------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :----------------------------------------------------------------------: |
| [DeepFillv2](/configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba.py) | free-form |  256x256   |     20k     | CelebA-val |  5.411   | 25.721 | 0.871 | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.log.json) |

## Citation

```bibtex
@inproceedings{yu2019free,
  title={Free-form image inpainting with gated convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4471--4480},
  year={2019}
}
```
