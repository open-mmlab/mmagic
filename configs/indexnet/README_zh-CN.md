# IndexNet (ICCV'2019)

> **任务**: 图像抠图

<!-- [ALGORITHM] -->

<details>
<summary align="right">IndexNet (ICCV'2019)</summary>

```bibtex
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

</details>

<br/>

|                          算法                           |   SAD    |    MSE    |   GRAD   |   CONN   | GPU 信息 |                                                   下载                                                   |
| :-----------------------------------------------------: | :------: | :-------: | :------: | :------: | :------: | :------------------------------------------------------------------------------------------------------: |
|                     M2O DINs (原文)                     |   45.8   |   0.013   |   25.9   | **43.7** |    -     |                                                    -                                                     |
| [M2O DINs (复现)](./indexnet_mobv2_1xb16-78k_comp1k.py) | **45.6** | **0.012** | **25.5** |   44.8   |    1     | [模型](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_20200618_173817.log.json) |

> The performance of training (best performance) with different random seeds diverges in a large range. You may need to run several experiments for each setting to obtain the above performance.

**其他结果**

|                                   算法                                    | SAD  |  MSE  | GRAD | CONN | GPU 信息 |                                                  下载                                                  |
| :-----------------------------------------------------------------------: | :--: | :---: | :--: | :--: | :------: | :----------------------------------------------------------------------------------------------------: |
| [M2O DINs (使用 DIM 流水线)](./indexnet_mobv2-dimaug_1xb16-78k_comp1k.py) | 50.1 | 0.016 | 30.8 | 49.5 |    1     | [模型](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k_SAD-50.1_20200626_231857-af359436.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k_20200626_231857.log.json) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py

# 单个GPU上训练
python tools/train.py configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py

# 多个GPU上训练
./tools/dist_train.sh configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth

# 单个GPU上测试
python tools/test.py configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth

# 多个GPU上测试
./tools/dist_test.sh configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
