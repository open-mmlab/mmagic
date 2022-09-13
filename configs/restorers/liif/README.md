# LIIF (CVPR'2021)

> [Learning Continuous Image Representation with Local Implicit Image Function](https://arxiv.org/abs/2012.09161)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

How to represent an image? While the visual world is presented in a continuous manner, machines store and see the images in a discrete way with 2D arrays of pixels. In this paper, we seek to learn a continuous representation for images. Inspired by the recent progress in 3D reconstruction with implicit neural representation, we propose Local Implicit Image Function (LIIF), which takes an image coordinate and the 2D deep features around the coordinate as inputs, predicts the RGB value at a given coordinate as an output. Since the coordinates are continuous, LIIF can be presented in arbitrary resolution. To generate the continuous representation for images, we train an encoder with LIIF representation via a self-supervised task with super-resolution. The learned continuous representation can be presented in arbitrary resolution even extrapolate to x30 higher resolution, where the training tasks are not provided. We further show that LIIF representation builds a bridge between discrete and continuous representation in 2D, it naturally supports the learning tasks with size-varied image ground-truths and significantly outperforms the method with resizing the ground-truths.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144032669-da59d683-9c4f-4598-a680-32770a369b74.png" width="400"/>
</div >

## Results and models

|                               Method                               | scale | Set5<br>PSNR / SSIM | Set14<br>PSNR / SSIM | DIV2K <br>PSNR / SSIM |                               Download                                |
| :----------------------------------------------------------------: | :---: | :-----------------: | :------------------: | :-------------------: | :-------------------------------------------------------------------: |
| [liif_edsr_norm_c64b16_g1_1000k_div2k](/configs/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k.py) |  x2   |  35.7131 / 0.9366   |   31.5579 / 0.8889   |   34.6647 / 0.9355    | [model](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.log.json) |
|                                 △                                  |  x3   |  32.3805 / 0.8915   |   28.4605 / 0.8039   |   30.9808 / 0.8724    |                                   △                                   |
|                                 △                                  |  x4   |  30.2748 / 0.8509   |   26.8415 / 0.7381   |   29.0245 / 0.8187    |                                   △                                   |
|                                 △                                  |  x6   |  27.1187 / 0.7774   |   24.7461 / 0.6444   |   26.7770 / 0.7425    |                                   △                                   |
|                                 △                                  |  x18  |  20.8516 / 0.5406   |   20.0096 / 0.4525   |   22.1987 / 0.5955    |                                   △                                   |
|                                 △                                  |  x30  |  18.8467 / 0.5010   |   18.1321 / 0.3963   |   20.5050 / 0.5577    |                                   △                                   |
| [liif_rdn_norm_c64b16_g1_1000k_div2k](/configs/restorers/liif/liif_rdn_norm_x2-4_c64b16_g1_1000k_div2k.py) |  x2   |  35.7874 / 0.9366   |   31.6866 / 0.8896   |   34.7548 / 0.9356    | [model](https://download.openmmlab.com/mmediting/restorers/liif/liif_rdn_norm_c64b16_g1_1000k_div2k_20210717-22d6fdc8.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/liif/liif_rdn_norm_c64b16_g1_1000k_div2k_20210717-22d6fdc8.log.json) |
|                                 △                                  |  x3   |  32.4992 / 0.8923   |   28.4905 / 0.8037   |   31.0744 / 0.8731    |                                   △                                   |
|                                 △                                  |  x4   |  30.3835 / 0.8513   |   26.8734 / 0.7373   |   29.1101 / 0.8197    |                                   △                                   |
|                                 △                                  |  x6   |  27.1914 / 0.7751   |   24.7824 / 0.6434   |   26.8693 / 0.7437    |                                   △                                   |
|                                 △                                  |  x18  |  20.8913 / 0.5329   |   20.1077 / 0.4537   |   22.2972 / 0.5950    |                                   △                                   |
|                                 △                                  |  x30  |  18.9354 / 0.4864   |   18.1448 / 0.3942   |   20.5663 / 0.5560    |                                   △                                   |

Note:

- △ refers to ditto.
- Evaluated on RGB channels,  `scale` pixels in each border are cropped before evaluation.

## Citation

```bibtex
@inproceedings{chen2021learning,
  title={Learning continuous image representation with local implicit image function},
  author={Chen, Yinbo and Liu, Sifei and Wang, Xiaolong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8628--8638},
  year={2021}
}
```
