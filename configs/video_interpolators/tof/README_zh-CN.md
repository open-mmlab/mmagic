# TOFlow (IJCV'2019)

> [Video Enhancement with Task-Oriented Flow](https://arxiv.org/abs/1711.09078)

<!-- [ALGORITHM] -->

## 预训练模型测试结果

在 RGB 通道上评估。
评估指标 `PSNR / SSIM`。

|                                                                 算法                                                                 |                                                                  预训练 SPyNet                                                                  | Vimeo90k-triplet |      GPU 信息       |                                                                                                                                                   下载                                                                                                                                                    |
| :----------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: | :-----------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        [tof_vfi_spynet_chair_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k.py)        |    [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth)     | 33.3294 / 0.9465 | 1 (Tesla PG503-216) |        [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.log.json)        |
|        [tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k.py)        |    [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_kitti_20220321-dbcc1cc1.pth)     | 33.3339 / 0.9466 | 1 (Tesla PG503-216) |        [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.log.json)        |
| [tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_clean_20220321-0756630b.pth) | 33.3170 / 0.9464 | 1 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.log.json) |
| [tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_final_20220321-5e89dcec.pth) | 33.3237 / 0.9465 | 1 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.log.json) |
|     [tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k](/configs/video_interpolators/tof/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k.py)     |   [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_pytoflow_20220321-5bab842d.pth)   | 33.3426 / 0.9467 | 1 (Tesla PG503-216) |     [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.log.json)     |

注: 由于 `batch_size=1` 预训练的 SPyNet 不包含 BN 层，这与 `https://github.com/Coldog2333/pytoflow` 一致.

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
