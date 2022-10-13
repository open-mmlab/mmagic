# PConv (ECCV'2018)

> [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Existing deep learning based image inpainting methods use a standard convolutional network over the corrupted image, using convolutional filter responses conditioned on both valid pixels as well as the substitute values in the masked holes (typically the mean value). This often leads to artifacts such as color discrepancy and blurriness. Post-processing is usually used to reduce such artifacts, but are expensive and may fail. We propose the use of partial convolutions, where the convolution is masked and renormalized to be conditioned on only valid pixels. We further include a mechanism to automatically generate an updated mask for the next layer as part of the forward pass. Our model outperforms other methods for irregular masks. We show qualitative and quantitative comparisons with other methods to validate our approach.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175613-1bc9ad1b-072d-4c1f-a97d-1af5be2590bd.png" width="400"/>
</div >

## Results and models

**Places365-Challenge**

|                                Method                                | Mask Type | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  |                                Download                                |
| :------------------------------------------------------------------: | :-------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :--------------------------------------------------------------------: |
| [PConv](/configs/inpainting/partial_conv/pconv_256x256_stage2_4x2_places.py) | free-form |  256x256   |    500k     | Places365-val |  8.776   | 22.762 | 0.801 | [model](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.log.json) |

**CelebA-HQ**

|                                Method                                 | Mask Type | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  |                                 Download                                 |
| :-------------------------------------------------------------------: | :-------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :----------------------------------------------------------------------: |
| [PConv](/configs/inpainting/partial_conv/pconv_256x256_stage2_4x2_celeba.py) | free-form |  256x256   |    500k     | CelebA-val |  5.990   | 25.404 | 0.853 | [model](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.log.json) |

## Citation

```bibtex
@inproceedings{liu2018image,
  title={Image inpainting for irregular holes using partial convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={85--100},
  year={2018}
}
```
