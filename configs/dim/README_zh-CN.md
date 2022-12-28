# DIM (CVPR'2017)

> **任务**: 图像抠图

<!-- [ALGORITHM] -->

<details>
<summary align="right">DIM (CVPR'2017)</summary>

```bibtex
@inproceedings{xu2017deep,
  title={Deep image matting},
  author={Xu, Ning and Price, Brian and Cohen, Scott and Huang, Thomas},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2970--2979},
  year={2017}
}
```

</details>

<br/>

|                                     算法                                      |   SAD    |    MSE    |   GRAD   |   CONN   | GPU 信息 |                                        下载                                        |
| :---------------------------------------------------------------------------: | :------: | :-------: | :------: | :------: | :------: | :--------------------------------------------------------------------------------: |
|                                第一阶段 (原文)                                |   54.6   |   0.017   |   36.7   |   55.3   |    -     |                                         -                                          |
|                                第三阶段 (原文)                                | **50.4** | **0.014** |   31.0   |   50.8   |    -     |                                         -                                          |
|           [第一阶段 (复现)](./dim_stage1-v16_1xb1-1000k_comp1k.py)            |   53.8   |   0.017   |   32.7   |   54.5   |    1     | [模型](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage1_v16_1x1_1000k_comp1k_SAD-53.8_20200605_140257-979a420f.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage1_v16_1x1_1000k_comp1k_20200605_140257.log.json) |
|         [第二阶段 (复现)](./dim_stage2-v16-pln_1xb1-1000k_comp1k.py)          |   52.3   |   0.016   |   29.4   |   52.4   |    1     | [模型](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage2_v16_pln_1x1_1000k_comp1k_20200607_171909.log.json) |
|         [第三阶段 (复现)](./dim_stage3-v16-pln_1xb1-1000k_comp1k.py)          |   50.6   |   0.015   | **29.0** | **50.7** |    1     | [模型](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_20200609_111851.log.json) |
| [第一阶段 (online merge)](./dim_stage1-v16_1xb1-1000k_comp1k_online-merge.py) |    -     |     -     |    -     |    -     |    -     |                                         -                                          |

**注**

- 第一阶段：训练不带精炼器的编码器-解码器部分。 \\
- 第二阶段：固定编码器-解码器部分，训练精炼器部分。 \\
- 第三阶段：微调整个网络模型。

> 模型在训练过程中的性能不稳定。因此，展示的性能并非来自最后一个模型权重文件，而是训练期间在验证集上取得的最佳性能。

> 不同随机种子的训练性能（最佳性能）的发散程度很大，您可能需要为每个设置运行多个实验以获得上述性能。

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

DIM 训练有三个阶段。

**阶段 1**: 训练不带精炼器的编码器-解码器部分。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/dim/dim_stage1-v16_1xb1-1000k_comp1k.py

# 单个GPU上训练
python tools/train.py configs/dim/dim_stage1-v16_1xb1-1000k_comp1k.py

# 多个GPU上训练
./tools/dist_train.sh configs/dim/dim_stage1-v16_1xb1-1000k_comp1k.py 8
```

**阶段 2**: 固定编码器-解码器部分，训练精炼器部分。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/dim/dim_stage2-v16-pln_1xb1-1000k_comp1k.py

# 单个GPU上训练
python tools/train.py configs/dim/dim_stage2-v16-pln_1xb1-1000k_comp1k.py

# 多个GPU上训练
./tools/dist_train.sh configs/dim/dim_stage2-v16-pln_1xb1-1000k_comp1k.py 8
```

**阶段 3**: 微调整个网络模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py

# 单个GPU上训练
python tools/train.py configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py

# 多个GPU上训练
./tools/dist_train.sh configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py onfigs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth

# 单个GPU上测试
python tools/test.py onfigs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth

# 多个GPU上测试
./tools/dist_test.sh onfigs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
