# DeepFillv1 (CVPR'2018)

> **任务**: 图像修复

<!-- [ALGORITHM] -->

<details>
<summary align="right">DeepFillv1 (CVPR'2018)</summary>

```bibtex
@inproceedings{yu2018generative,
  title={Generative image inpainting with contextual attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5505--5514},
  year={2018}
}
```

</details>

<br/>

**CelebA-HQ**

|                       算法                        |  掩膜类型   | 分辨率  | 训练集容量 |   测试集   | l1 损失 |  PSNR  | SSIM  | GPU 信息 |                                          下载                                           |
| :-----------------------------------------------: | :---------: | :-----: | :--------: | :--------: | :-----: | :----: | :---: | :------: | :-------------------------------------------------------------------------------------: |
| [DeepFillv1](./deepfillv1_4xb4_celeba-256x256.py) | square bbox | 256x256 |   1500k    | CelebA-val |  6.677  | 26.878 | 0.911 |    4     | [模型](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.log.json) |

**Places365-Challenge**

|                       算法                        |  掩膜类型   | 分辨率  | 训练集容量 |    测试集     | l1 损失 |  PSNR  | SSIM  | GPU 信息 |                                         下载                                         |
| :-----------------------------------------------: | :---------: | :-----: | :--------: | :-----------: | :-----: | :----: | :---: | :------: | :----------------------------------------------------------------------------------: |
| [DeepFillv1](./deepfillv1_8xb2_places-256x256.py) | square bbox | 256x256 |   3500k    | Places365-val | 11.019  | 23.429 | 0.862 |    8     | [模型](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/deepfillv1/deepfillv1_8xb2_places-256x256.py

# 单个GPU上训练
python tools/train.py configs/deepfillv1/deepfillv1_8xb2_places-256x256.py

# 多个GPU上训练
./tools/dist_train.sh configs/deepfillv1/deepfillv1_8xb2_places-256x256.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/deepfillv1/deepfillv1_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth

# 单个GPU上测试
python tools/test.py configs/deepfillv1/deepfillv1_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth

# 多个GPU上测试
./tools/dist_test.sh configs/deepfillv1/deepfillv1_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
