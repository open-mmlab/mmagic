# SRCNN (TPAMI'2015)

> [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN) that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional sparse-coding-based SR methods can also be viewed as a deep convolutional network. But unlike traditional methods that handle each component separately, our method jointly optimizes all layers. Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. We explore different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144034831-79f48aae-196e-42e7-92b9-069149733e3e.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                   Method                                   |       Set5       |       Set14       |      DIV2K       |                                   Download                                    |
| :------------------------------------------------------------------------: | :--------------: | :---------------: | :--------------: | :---------------------------------------------------------------------------: |
| [srcnn_x4k915_1x16_1000k_div2k](/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py) | 28.4316 / 0.8099 | 25.6486 /  0.7014 | 27.7460 / 0.7854 | [model](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608_120159.log.json) |

## Citation

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
