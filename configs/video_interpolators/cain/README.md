# CAIN (AAAI'2020)

> [Channel Attention Is All You Need for Video Frame Interpolation](https://aaai.org/ojs/index.php/AAAI/article/view/6693/6547)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Prevailing video frame interpolation techniques rely heavily on optical flow estimation and require additional model complexity and computational cost; it is also susceptible to error propagation in challenging scenarios with large motion and heavy occlusion. To alleviate the limitation, we propose a simple but effective deep neural network for video frame interpolation, which is end-to-end trainable and is free from a motion estimation network component. Our algorithm employs a special feature reshaping operation, referred to as PixelShuffle, with a channel attention, which replaces the optical flow computation module. The main idea behind the design is to distribute the information in a feature map into multiple channels and extract motion information by attending the channels for pixel-level frame synthesis. The model given by this principle turns out to be effective in the presence of challenging motion and occlusion. We construct a comprehensive evaluation benchmark and demonstrate that the proposed approach achieves outstanding performance compared to the existing models with a component for optical flow computation.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/56712176/149734064-1da0cebf-6953-4106-a29a-43acd7386a80.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .
The learning rate adjustment strategy is `Step LR scheduler with min_lr clipping`.

|                                           Method                                           | vimeo-90k-triplet |                                           Download                                            |
| :----------------------------------------------------------------------------------------: | :---------------: | :-------------------------------------------------------------------------------------------: |
| [cain_b5_g1b32_vimeo90k_triplet](/configs/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet.py) | 34.6010 / 0.9578  | [model](https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.pth)/[log](https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.log.json) |

## Citation

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
