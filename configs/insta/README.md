# Instance-aware Image Colorization (CVPR'2020)

> **任务**: 图像上色

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/insta/insta_full_cocostuff_256x256.py

# 单个GPU上训练
python tools/train.py configs/insta/insta_full_cocostuff_256x256.py

# 多个GPU上训练
./tools/dist_train.sh configs/insta/insta_full_cocostuff_256x256.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python demo/colorization_demo.py configs/insta/insta_full_cocostuff_256x256.py ../checkpoints/instance_aware_cocostuff.pth

# 单个GPU上测试
python demo/colorization_demo.py configs/insta/insta_full_cocostuff_256x256.py ../checkpoints/instance_aware_cocostuff.pth

# 多个GPU上测试
./tools/dist_test.sh configs/insta/insta_full_cocostuff_256x256.py ../checkpoints/instance_aware_cocostuff.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>


<details>
<summary align="right">Instance-aware Image Colorization (CVPR'2020)</summary>

```bibtex
@inproceedings{Su-CVPR-2020,
  author = {Su, Jheng-Wei and Chu, Hung-Kuo and Huang, Jia-Bin},
  title = {Instance-aware Image Colorization},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

</details>

