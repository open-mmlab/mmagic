# FLAVR (arXiv'2020)

> [FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation](https://arxiv.org/abs/2012.08512.pdf)

> **任务**: 视频插帧

<!-- [ALGORITHM] -->

## 预训练模型测试结果

在 RGB 通道上评估。
评估指标 `PSNR / SSIM`。

|                                      算法                                       | scale | Vimeo90k-triplet  |      GPU 信息       |                                       下载                                       |
| :-----------------------------------------------------------------------------: | :---: | :---------------: | :-----------------: | :------------------------------------------------------------------------------: |
| [flavr_in4out1_g8b4_vimeo90k_septuplet](./flavr_in4out1_8xb4_vimeo90k-septuplet.py) |  x2   | 36.3340 / 0.96015 | 8 (Tesla PG503-216) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.log.json) |

注：FLAVR 中的 8 倍视频插帧算法将会在未来版本中支持。

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py

# 单个GPU上训练
python tools/train.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py

# 多个GPU上训练
./tools/dist_train.sh configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth

# 单个GPU上测试
python tools/test.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth

# 多个GPU上测试
./tools/dist_test.sh configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>

## Citation

```bibtex
@article{kalluri2020flavr,
  title={Flavr: Flow-agnostic video representations for fast frame interpolation},
  author={Kalluri, Tarun and Pathak, Deepak and Chandraker, Manmohan and Tran, Du},
  journal={arXiv preprint arXiv:2012.08512},
  year={2020}
}
```
