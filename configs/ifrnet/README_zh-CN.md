# IFRNet (CVPR'2022)

> **任务**: 视频插帧

<!-- [ALGORITHM] -->

<details>
<summary align="right">IFRNet (CVPR'2022)</summary>

```bibtex
@InProceedings{Kong_2022_CVPR,
  author = {Kong, Lingtong and Jiang, Boyuan and Luo, Donghao and Chu, Wenqing and Huang, Xiaoming and Tai, Ying and Wang, Chengjie and Yang, Jie},
  title = {IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```

</details>

|                            方法                             | 插帧比率 |      PSNR      |      SSIM      | GPU信息  |                                               下载                                                |
| :---------------------------------------------------------: | :------: | :------------: | :------------: | :------: | :-----------------------------------------------------------------------------------------------: |
| [ifrnet_in2out1_8xb4_vimeo](./ifrnet_in2out1_8xb4_vimeo.py) |    x2    | 35.7999(35.80) | 0.9680(0.9794) | 1 (A100) | [模型](https://download.openmmlab.com/mmediting/ifrnet/IFRNet_vimeo90k-7a66b214.pth) \| 日志（即将到来） |
| [ifrnet_in2out7_8xb4_gopro](./ifrnet_in2out7_8xb4_gopro.py) |    x8    | 29.9394(29.84) | 0.8922(0.920)  | 1 (A100) | [模型](https://download.openmmlab.com/mmediting/ifrnet/IFRNet_gopro-5d2f805a.pth) \| 日志（即将到来） |
| [ifrnet_in2out7_8xb4_adobe](./ifrnet_in2out7_8xb4_adobe.py) |    x8    | 30.0273(31.93) | 0.9057(0.943)  | 1 (A100) |                                模型与上述一致 \| 日志（即将到来）                                 |
|                            Note:                            |          |                |                |          |                                                                                                   |

- 评估结果a(b)中，a代表由MMEditing测量，b代表由原论文提供。
- PSNR是在RGB通道评估。
- SSIM是平均的分别在RGB通道评估的SSIM, 而原论文使用了3D的SSIM卷积核做统一评估。
- 测评的图像是在截取了原图像正中间大小为 512 X 512 的图片。
- 因为缺少原论文中测评Adobe240fps数据集所使用的视频目录，MMEditing测评中使用了以下这些视频：*720p_240fps_1*, *GOPR9635*, *GOPR9637a*, *IMG_0004a*, *IMG_0015*, *IMG_0023*, *IMG_0179*, *IMG_0183*。

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

可用于训练的模型不久会更新。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/ifrnet/ifrnet_in2out7_8xb4_gopro.py /path/to/checkpoint

# 单个GPU上测试
python tools/test.py configs/ifrnet/ifrnet_in2out7_8xb4_gopro.py /path/to/checkpoint

# 多个GPU上测试
./tools/dist_test.sh configs/ifrnet/ifrnet_in2out7_8xb4_gopro.py /path/to/checkpoint 8
```

预训练模型未来将会上传，敬请等待。
更多细节可以参考 [train_test.md](../../docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
