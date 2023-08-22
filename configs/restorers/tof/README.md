# TOFlow (IJCV'2019)

> [Video Enhancement with Task-Oriented Flow](https://arxiv.org/abs/1711.09078)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Many video enhancement algorithms rely on optical flow to register frames in a video sequence. Precise flow estimation is however intractable; and optical flow itself is often a sub-optimal representation for particular video processing tasks. In this paper, we propose task-oriented flow (TOFlow), a motion representation learned in a self-supervised, task-specific manner. We design a neural network with a trainable motion estimation component and a video processing component, and train them jointly to learn the task-oriented flow. For evaluation, we build Vimeo-90K, a large-scale, high-quality video dataset for low-level video processing. TOFlow outperforms traditional optical flow on standard benchmarks as well as our Vimeo-90K dataset in three video processing tasks: frame interpolation, video denoising/deblocking, and video super-resolution.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035477-2480d580-1409-4a7c-88d5-c13a3dbd62ac.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                     Method                                     |       Vid4       |                                               Download                                                |
| :----------------------------------------------------------------------------: | :--------------: | :---------------------------------------------------------------------------------------------------: |
| [tof_x4_vimeo90k_official](/configs/restorers/tof/tof_x4_vimeo90k_official.py) | 24.4377 / 0.7433 | [model](https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth) |

## Citation

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
