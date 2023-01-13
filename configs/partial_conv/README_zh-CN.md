# PConv (ECCV'2018)

> **任务**: 图像修复

<!-- [ALGORITHM] -->

<details>
<summary align="right">PConv (ECCV'2018)</summary>

```bibtex
@inproceedings{liu2018image,
  title={Image inpainting for irregular holes using partial convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={85--100},
  year={2018}
}
```

</details>

<br/>

**Places365-Challenge**

|                          算法                          | 掩膜类型  | 分辨率  | 训练集容量 |    测试集     | l1 损失 |  PSNR  | SSIM  | GPU 信息 |                                       下载                                        |
| :----------------------------------------------------: | :-------: | :-----: | :--------: | :-----------: | :-----: | :----: | :---: | :------: | :-------------------------------------------------------------------------------: |
| [PConv_Stage1](./pconv_stage1_8xb12_places-256x256.py) | free-form | 256x256 |    500k    | Places365-val |    -    |   -    |   -   |    8     |                                         -                                         |
| [PConv_Stage2](./pconv_stage2_4xb2_places-256x256.py)  | free-form | 256x256 |    500k    | Places365-val |  8.776  | 22.762 | 0.801 |    4     | [模型](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.log.json) |

**CelebA-HQ**

|                         算法                          | 掩膜类型  | 分辨率  | 训练集容量 |   测试集   | l1 损失 |  PSNR  | SSIM  | GPU 信息 |                                         下载                                          |
| :---------------------------------------------------: | :-------: | :-----: | :--------: | :--------: | :-----: | :----: | :---: | :------: | :-----------------------------------------------------------------------------------: |
| [PConv_Stage1](./pconv_stage1_8xb1_celeba-256x256.py) | free-form | 256x256 |    500k    | CelebA-val |    -    |   -    |   -   |    8     |                                           -                                           |
| [PConv_Stage2](./pconv_stage2_4xb2_celeba-256x256.py) | free-form | 256x256 |    500k    | CelebA-val |  5.990  | 25.404 | 0.853 |    4     | [模型](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/partial_conv/pconv_stage2_4xb2_places-256x256.py

# 单个GPU上训练
python tools/train.py configs/partial_conv/pconv_stage2_4xb2_places-256x256.py

# 多个GPU上训练
./tools/dist_train.sh configs/partial_conv/pconv_stage2_4xb2_places-256x256.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/partial_conv/pconv_stage2_4xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth

# 单个GPU上测试
python tools/test.py configs/partial_conv/pconv_stage2_4xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth

# 多个GPU上测试
./tools/dist_test.sh configs/partial_conv/pconv_stage2_4xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
