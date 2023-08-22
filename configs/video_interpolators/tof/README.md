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

|                          Method                           |                          Pretrained SPyNet                           | Vimeo90k-triplet |                          Download                           |
| :-------------------------------------------------------: | :------------------------------------------------------------------: | :--------------: | :---------------------------------------------------------: |
| [tof_vfi_spynet_chair_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth) | 33.3294 / 0.9465 | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.log.json) |
| [tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_kitti_20220321-dbcc1cc1.pth) | 33.3339 / 0.9466 | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.log.json) |
| [tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_clean_20220321-0756630b.pth) | 33.3170 / 0.9464 | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.log.json) |
| [tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_final_20220321-5e89dcec.pth) | 33.3237 / 0.9465 | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.log.json) |
| [tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_pytoflow_20220321-5bab842d.pth) | 33.3426 / 0.9467 | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.log.json) |

Note: These pretrained SPyNets don't contain BN layer since `batch_size=1`, which is consistent with `https://github.com/Coldog2333/pytoflow`.

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
