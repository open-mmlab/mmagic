# CAIN (AAAI'2020)

> **任务**: 视频插帧

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://aaai.org/ojs/index.php/AAAI/article/view/6693/6547">CAIN (AAAI'2020)</a></summary>

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

</details>

<br/>

在 RGB 通道上进行评估。
我们使用 `PSNR` 和 `SSIM` 作为指标。
学习率调整策略是等间隔调整策略。

|                                  算法                                   | vimeo-90k-triplet |         GPU 信息         |                                           下载                                           |
| :---------------------------------------------------------------------: | :---------------: | :----------------------: | :--------------------------------------------------------------------------------------: |
| [cain_b5_g1b32_vimeo90k_triplet](./cain_g1b32_1xb5_vimeo90k-triplet.py) | 34.6010 / 0.9578  | 1 (Tesla V100-SXM2-32GB) | [模型](https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.pth)/[日志](https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py

# 单个GPU上训练
python tools/train.py configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py

# 多个GPU上训练
./tools/dist_train.sh configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.pth

# 单个GPU上测试
python tools/test.py configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.pth

# 多个GPU上测试
./tools/dist_test.sh configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
