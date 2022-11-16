# NAFNET (ECCV'2022)

> **任务**: 图像恢复

<!-- [ALGORITHM] -->

<details>
<summary align="right">NAFNET (ECCV'2022)</summary>

```bibtex
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
```

</details>

|                              方法                              | 图片尺寸 |    GoPro PSNR    |   GoPro SSIM   |    SIDD PSNR     |   SIDD SSIM    | GPU 信息 |                              下载                              |
| :------------------------------------------------------------: | :------: | :--------------: | :------------: | :--------------: | :------------: | :------: | :------------------------------------------------------------: |
| [nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd](/configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py) | 256X256  |        -         |       -        | 37.5855(40.3045) | 0.9095(0.9614) | 1 (A100) | [model](https://download.openmmlab.com/mmediting/nafnet/NAFNet-SIDD-midc64.pth) \| log(即将到来) |
| [nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro](/configs/nafnet/nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro.py) | 1280x720 | 33.7246(33.7103) | 0.9479(0.9668) |        -         |       -        | 1 (A100) | [model](https://download.openmmlab.com/mmediting/nafnet/NAFNet-GoPro-midc64.pth) \| log(即将到来) |

Note:

- 对于SIDD数据集，NAFNet中使用了 `lmdb`格式的数据集而这里我们使用的是从`lmdb`文件中提取的`图片`数据集。
- PSNR是在RGB通道评估。
- SSIM是平均的分别在RGB通道评估的SSIM, 而原论文使用了3D的SSIM卷积核做统一评估。

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py

# 单个GPU上训练
python tools/train.py configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py

# 多个GPU上训练
./tools/dist_train.sh configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py /path/to/checkpoint

# 单个GPU上测试
python tools/test.py configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py /path/to/checkpoint

# 多个GPU上测试
./tools/dist_test.sh configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py /path/to/checkpoint 8
```

预训练模型未来将会上传，敬请等待。
更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
