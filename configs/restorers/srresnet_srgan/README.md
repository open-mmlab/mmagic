# SRGAN (CVPR'2016)

> [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035016-8ed4a80b-2d8b-4947-848b-3f8e917a9273.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.

The metrics are `PSNR / SSIM` .

|                                   Method                                   |       Set5        |      Set14       |      DIV2K       |                                   Download                                    |
| :------------------------------------------------------------------------: | :---------------: | :--------------: | :--------------: | :---------------------------------------------------------------------------: |
| [msrresnet_x4c64b16_1x16_300k_div2k](/configs/restorers/srresnet_srgan/msrresnet_x4c64b16_g1_1000k_div2k.py) | 30.2252 / 0.8491  | 26.7762 / 0.7369 | 28.9748 / 0.8178 | [model](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521-61556be5.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/msrresnet_x4c64b16_1x16_300k_div2k_20200521_110246.log.json) |
| [srgan_x4c64b16_1x16_1000k_div2k](/configs/restorers/srresnet_srgan/srgan_x4c64b16_g1_1000k_div2k.py) | 27.9499 /  0.7846 | 24.7383 / 0.6491 | 26.5697 / 0.7365 | [model](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200606-a1f0810e.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srresnet_srgan/srgan_x4c64b16_1x16_1000k_div2k_20200506_191442.log.json) |

## Citation

```bibtex
@inproceedings{ledig2016photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  year={2016}
}
```
