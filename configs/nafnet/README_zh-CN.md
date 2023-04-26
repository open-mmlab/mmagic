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

|                                     方法                                      | 图片尺寸 |       PSNR       |      SSIM      | GPU信息  |                                     下载                                      |
| :---------------------------------------------------------------------------: | :------: | :--------------: | :------------: | :------: | :---------------------------------------------------------------------------: |
| [nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd](./nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py) | 256X256  | 40.3045(40.3045) | 0.9253(0.9614) | 1 (A100) | [模型](https://download.openmmlab.com/mmediting/nafnet/NAFNet-SIDD-midc64.pth) \| 日志(即将到来) |
| [nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro](./nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro.py) | 1280x720 | 33.7246(33.7103) | 0.9479(0.9668) | 1 (A100) | [模型](https://download.openmmlab.com/mmediting/nafnet/NAFNet-GoPro-midc64.pth) \| 日志(即将到来) |

Note:

- 评估结果a(b)中，a代表由MMagic测量，b代表由原论文提供。
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

更多细节可以参考 [train_test.md](../../docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

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
更多细节可以参考 [train_test.md](../../docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
