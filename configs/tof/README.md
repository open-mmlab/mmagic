# TOFlow (IJCV'2019)

> [Video Enhancement with Task-Oriented Flow](https://arxiv.org/abs/1711.09078)

> **Task**: Video Interpolation, Video Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Many video enhancement algorithms rely on optical flow to register frames in a video sequence. Precise flow estimation is however intractable; and optical flow itself is often a sub-optimal representation for particular video processing tasks. In this paper, we propose task-oriented flow (TOFlow), a motion representation learned in a self-supervised, task-specific manner. We design a neural network with a trainable motion estimation component and a video processing component, and train them jointly to learn the task-oriented flow. For evaluation, we build Vimeo-90K, a large-scale, high-quality video dataset for low-level video processing. TOFlow outperforms traditional optical flow on standard benchmarks as well as our Vimeo-90K dataset in three video processing tasks: frame interpolation, video denoising/deblocking, and video super-resolution.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035477-2480d580-1409-4a7c-88d5-c13a3dbd62ac.png" width="400"/>
</div >

## Results and models

Evaluated on Vimeo90k-triplet (RGB channels).
The metrics are `PSNR / SSIM` .

|                   Model                   |     Dataset      |        Task         |                   Pretrained SPyNet                   |  PSNR   | Training Resources  |                   Download                    |
| :---------------------------------------: | :--------------: | :-----------------: | :---------------------------------------------------: | :-----: | :-----------------: | :-------------------------------------------: |
| [tof_vfi_spynet_chair_nobn_1xb1_vimeo90k](./tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Interpolation | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth) | 33.3294 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.log.json) |
| [tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k](./tof_spynet-kitti-wobn_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Interpolation | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_kitti_20220321-dbcc1cc1.pth) | 33.3339 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.log.json) |
| [tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k](./tof_spynet-sintel-wobn-clean_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Interpolation | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_clean_20220321-0756630b.pth) | 33.3170 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.log.json) |
| [tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k](./tof_spynet-sintel-wobn-final_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Interpolation | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_final_20220321-5e89dcec.pth) | 33.3237 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.log.json) |
| [tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k](./tof_spynet-pytoflow-wobn_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Interpolation | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_pytoflow_20220321-5bab842d.pth) | 33.3426 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.log.json) |

|                  Model                   |     Dataset      |          Task          |                   Pretrained SPyNet                   |  SSIM  | Training Resources  |                   Download                   |
| :--------------------------------------: | :--------------: | :--------------------: | :---------------------------------------------------: | :----: | :-----------------: | :------------------------------------------: |
| [tof_vfi_spynet_chair_nobn_1xb1_vimeo90k](./tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Super-Resolution | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth) | 0.9465 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.log.json) |
| [tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k](./tof_spynet-kitti-wobn_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Super-Resolution | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_kitti_20220321-dbcc1cc1.pth) | 0.9466 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.log.json) |
| [tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k](./tof_spynet-sintel-wobn-clean_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Super-Resolution | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_clean_20220321-0756630b.pth) | 0.9464 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.log.json) |
| [tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k](./tof_spynet-sintel-wobn-final_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Super-Resolution | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_final_20220321-5e89dcec.pth) | 0.9465 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.log.json) |
| [tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k](./tof_spynet-pytoflow-wobn_1xb1_vimeo90k-triplet.py) | Vimeo90k-triplet | Video Super-Resolution | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_pytoflow_20220321-5bab842d.pth) | 0.9467 | 1 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.log.json) |

Note: These pretrained SPyNets don't contain BN layer since `batch_size=1`, which is consistent with `https://github.com/Coldog2333/pytoflow`.

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                           Model                           | Dataset  |          Task          |       Vid4       | Training Resources |                                    Download                                     |
| :-------------------------------------------------------: | :------: | :--------------------: | :--------------: | :----------------: | :-----------------------------------------------------------------------------: |
| [tof_x4_vimeo90k_official](./tof_x4_official_vimeo90k.py) | vimeo90k | Video Super-Resolution | 24.4377 / 0.7433 |         -          | [model](https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

TOF only supports video interpolation task for training now.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py

# single-gpu train
python tools/train.py configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py

# multi-gpu train
./tools/dist_train.sh configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

TOF supports two tasks for testing.

**Task 1**: Video Interpolation

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth

# single-gpu test
python tools/test.py configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth

# multi-gpu test
./tools/dist_test.sh configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth 8
```

**Task 2**: Video Super-Resolution

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/tof/tof_x4_official_vimeo90k.py https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth

# single-gpu test
python tools/test.py configs/tof/tof_x4_official_vimeo90k.py https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth

# multi-gpu test
./tools/dist_test.sh configs/tof/tof_x4_official_vimeo90k.py https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

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
