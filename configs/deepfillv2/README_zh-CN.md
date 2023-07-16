# DeepFillv2 (CVPR'2019)

> **任务**: 图像修复

<!-- [ALGORITHM] -->

<details>
<summary align="right">DeepFillv2 (CVPR'2019)</summary>

```bibtex
@inproceedings{yu2019free,
  title={Free-form image inpainting with gated convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4471--4480},
  year={2019}
}
```

</details>

<br/>

**CelebA-HQ**

|                       算法                        | 掩膜类型  | 分辨率  | 训练集容量 |   测试集   | l1 损失 |  PSNR  | SSIM  | GPU 信息 |                                           下载                                            |
| :-----------------------------------------------: | :-------: | :-----: | :--------: | :--------: | :-----: | :----: | :---: | :------: | :---------------------------------------------------------------------------------------: |
| [DeepFillv2](./deepfillv2_8xb2_celeba-256x256.py) | free-form | 256x256 |    20k     | CelebA-val |  5.411  | 25.721 | 0.871 |    8     | [模型](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.log.json) |

**Places365-Challenge**

|                       算法                        | 掩膜类型  | 分辨率  | 训练集容量 |    测试集     | l1 损失 |  PSNR  | SSIM  | GPU 信息 |                                          下载                                          |
| :-----------------------------------------------: | :-------: | :-----: | :--------: | :-----------: | :-----: | :----: | :---: | :------: | :------------------------------------------------------------------------------------: |
| [DeepFillv2](./deepfillv2_8xb2_places-256x256.py) | free-form | 256x256 |    100k    | Places365-val |  8.635  | 22.398 | 0.815 |    8     | [模型](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/deepfillv2/deepfillv2_8xb2_places-256x256.py

# 单个GPU上训练
python tools/train.py configs/deepfillv2/deepfillv2_8xb2_places-256x256.py

# 多个GPU上训练
./tools/dist_train.sh configs/deepfillv2/deepfillv2_8xb2_places-256x256.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/deepfillv2/deepfillv2_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth

# 单个GPU上测试
python tools/test.py configs/deepfillv2/deepfillv2_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth

# 多个GPU上测试
./tools/dist_test.sh configs/deepfillv2/deepfillv2_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
