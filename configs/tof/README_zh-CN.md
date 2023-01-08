# TOFlow (IJCV'2019)

> [Video Enhancement with Task-Oriented Flow](https://arxiv.org/abs/1711.09078)

> **任务**: 视频插帧, 视频超分辨率

<!-- [ALGORITHM] -->

## 预训练模型测试结果

在 RGB 通道上评估。
评估指标 `PSNR / SSIM`。

|                         算法                         |                        预训练 SPyNet                         | Vimeo90k-triplet |      GPU 信息       |                         下载                          |
| :--------------------------------------------------: | :----------------------------------------------------------: | :--------------: | :-----------------: | :---------------------------------------------------: |
| [tof_vfi_spynet_chair_nobn_1xb1_vimeo90k](./tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth) | 33.3294 / 0.9465 | 1 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_chair_nobn_1xb1_vimeo90k_20220321-2fc9e258.log.json) |
| [tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k](./tof_spynet-kitti-wobn_1xb1_vimeo90k-triplet.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_kitti_20220321-dbcc1cc1.pth) | 33.3339 / 0.9466 | 1 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_kitti_nobn_1xb1_vimeo90k_20220321-3f7ca4cd.log.json) |
| [tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k](./tof_spynet-sintel-wobn-clean_1xb1_vimeo90k-triplet.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_clean_20220321-0756630b.pth) | 33.3170 / 0.9464 | 1 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_clean_nobn_1xb1_vimeo90k_20220321-6e52a6fd.log.json) |
| [tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k](./tof_spynet-sintel-wobn-final_1xb1_vimeo90k-triplet.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_sintel_final_20220321-5e89dcec.pth) | 33.3237 / 0.9465 | 1 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_sintel_final_nobn_1xb1_vimeo90k_20220321-8ab70dbb.log.json) |
| [tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k](./tof_spynet-pytoflow-wobn_1xb1_vimeo90k-triplet.py) | [spynet_chairs_final](https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_pytoflow_20220321-5bab842d.pth) | 33.3426 / 0.9467 | 1 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/toflow/tof_vfi_spynet_pytoflow_nobn_1xb1_vimeo90k_20220321-5f4b243e.log.json) |

注: 由于 `batch_size=1` 预训练的 SPyNet 不包含 BN 层，这与 `https://github.com/Coldog2333/pytoflow` 一致.

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

TOF 的训练仅支持视频插帧任务。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py

# 单个GPU上训练
python tools/train.py configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py

# 多个GPU上训练
./tools/dist_train.sh configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

TOF 的测试支持视频插帧和视频超分辨率两种任务。

**任务 1**: 视频插帧

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth

# 单个GPU上测试
python tools/test.py configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth

# 多个GPU上测试
./tools/dist_test.sh configs/tof/tof_spynet-chair-wobn_1xb1_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/toflow/pretrained_spynet_chair_20220321-4d82e91b.pth 8
```

**任务 2**: 视频超分辨率

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/tof/tof_x4_official_vimeo90k.py https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth

# 单个GPU上测试
python tools/test.py configs/tof/tof_x4_official_vimeo90k.py https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth

# 多个GPU上测试
./tools/dist_test.sh configs/tof/tof_x4_official_vimeo90k.py https://download.openmmlab.com/mmediting/restorers/tof/tof_x4_vimeo90k_official-a569ff50.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

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
