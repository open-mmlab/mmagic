# EDSR (CVPR'2017)

> [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Recent research on super-resolution has progressed with the development of deep convolutional neural networks (DCNN). In particular, residual learning techniques exhibit improved performance. In this paper, we develop an enhanced deep super-resolution network (EDSR) with performance exceeding those of current state-of-the-art SR methods. The significant performance improvement of our model is due to optimization by removing unnecessary modules in conventional residual networks. The performance is further improved by expanding the model size while we stabilize the training procedure. We also propose a new multi-scale deep super-resolution system (MDSR) and training method, which can reconstruct high-resolution images of different upscaling factors in a single model. The proposed methods show superior performance over the state-of-the-art methods on benchmark datasets and prove its excellence by winning the NTIRE2017 Super-Resolution Challenge.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144018090-ed629948-bf68-43ff-b2a9-6213e23f19a5.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                   Method                                    |       Set5       |      Set14       |      DIV2K       |                                   Download                                    |
| :-------------------------------------------------------------------------: | :--------------: | :--------------: | :--------------: | :---------------------------------------------------------------------------: |
| [edsr_x2c64b16_1x16_300k_div2k](/configs/restorers/edsr/edsr_x2c64b16_g1_300k_div2k.py) | 35.7592 / 0.9372 | 31.4290 / 0.8874 | 34.5896 / 0.9352 | [model](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x2c64b16_1x16_300k_div2k_20200604-19fe95ea.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x2c64b16_1x16_300k_div2k_20200604_221933.log.json) |
| [edsr_x3c64b16_1x16_300k_div2k](/configs/restorers/edsr/edsr_x3c64b16_g1_300k_div2k.py) | 32.3301 / 0.8912 | 28.4125 / 0.8022 | 30.9154 / 0.8711 | [model](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x3c64b16_1x16_300k_div2k_20200608-36d896f4.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x3c64b16_1x16_300k_div2k_20200608_114850.log.json) |
| [edsr_x4c64b16_1x16_300k_div2k](/configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k.py) | 30.2223 / 0.8500 | 26.7870 / 0.7366 | 28.9675 / 0.8172 | [model](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608-3c2af8a3.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edsr/edsr_x4c64b16_1x16_300k_div2k_20200608_115148.log.json) |

## Citation

```bibtex
@inproceedings{lim2017enhanced,
  title={Enhanced deep residual networks for single image super-resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={136--144},
  year={2017}
}
```
