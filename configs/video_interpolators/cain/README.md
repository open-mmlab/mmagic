# CAIN (AAAI'2020)

## Abstract

<!-- [ABSTRACT] -->

Prevailing video frame interpolation techniques rely heavily on optical flow estimation and require additional model complexity and computational cost; it is also susceptible to error propagation in challenging scenarios with large motion and heavy occlusion. To alleviate the limitation, we propose a simple but effective deep neural network for video frame interpolation, which is end-to-end trainable and is free from a motion estimation network component. Our algorithm employs a special feature reshaping operation, referred to as PixelShuffle, with a channel attention, which replaces the optical flow computation module. The main idea behind the design is to distribute the information in a feature map into multiple channels and extract motion information by attending the channels for pixel-level frame synthesis. The model given by this principle turns out to be effective in the presence of challenging motion and occlusion. We construct a comprehensive evaluation benchmark and demonstrate that the proposed approach achieves outstanding performance compared to the existing models with a component for optical flow computation.

<!-- [IMAGE] -->
<p align="center">
  <img src="https://user-images.githubusercontent.com/7676947/146704029-58bc4db4-267f-4158-8129-e49ab6652249.png" />
</p>

<!-- [PAPER_TITLE: Channel Attention Is All You Need for Video Frame Interpolation] -->
<!-- [PAPER_URL: https://aaai.org/ojs/index.php/AAAI/article/view/6693/6547] -->

## Citation

<!-- [ALGORITHM] -->

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

## Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                            Method                                           | vimeo-90k-triple |                                                                                                                                         Download                                                                                                                                         |
|:-------------------------------------------------------------------------------------------:|:----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [cain_b5_320k_vimeo-triple](/configs/video_interpolators/cain/cain_b5_320k_vimeo-triple.py) |   34.49/0.9565   | [model](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth)/[log](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104_183640.log.json) |
