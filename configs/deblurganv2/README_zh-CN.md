# DeblurGAN-v2 (ICCV'2019)

> [DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826)

> **任务**: 去模糊

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

我们提出了一种新的端到端的用于单图像运动去模糊生成对抗网络（GAN），名为DeblurGAN-v2，它显著提高了最高水平的去模糊效率、质量和灵活性。DeblurGAN-v2是基于具有双尺度鉴别器的相对论条件GAN。我们首次将特征金字塔网络引入去模糊来作为DeblurGAN-v2生成器的核心构建块。它可以灵活地与各种主干网络一起工作，以在性能和效率之间取得平衡。先进的主干网络的插件（例如，Inception-ResNet-v2）可以带来最先进的去模糊处理。同时，凭借轻量级主干网络（例如MobileNet及其变体），DeblurGAN-v2的速度比最接近的竞争对手快10-100倍，同时保持接近最先进的结果，这意味着可用于实时视频去模糊。我们证明了DeblurGAN-v2在几个流行的基准测试中获得了非常有竞争力的性能，包括去模糊质量（客观和主观）以及效率。此外，我们还展示了该架构对于一般图像恢复任务也依然有效。

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/master/doc_images/pipeline.jpg"/>
</div>

## 结果与模型

<div align="center">
  <b> DEBLURGANv2 256x256</b>
  <br/>
  <img src="https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/master/doc_images/kohler_visual.png" width="800"/>
  <img src="https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/master/doc_images/restore_visual.png" width="800"/>
 </div>

|                            算法                            |       测试集       |     生成器模型     | 判别器模型 | PSNR/<br/>SSIM |                                        下载                                        |
| :--------------------------------------------------------: | :----------------: | :----------------: | :--------: | :------------: | :--------------------------------------------------------------------------------: |
| [fpn_inception](./deblurganv2_fpn-inception_1xb1_gopro.py) | GoPro Test Dataset | InceptionResNet-v2 | double_gan |  29.55/ 0.934  | [模型](https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-inception.pth) \\ [日志](<>) |
| [fpn_mobilenet](./deblurganv2_fpn-mobilenet_1xb1_gopro.py) | GoPro Test Dataset |     MobileNet      | double_gan |  28.17/ 0.925  | [模型](https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-mobilenet.pth) \\ [日志](<>) |

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py

# single-gpu train
python tools/train.py configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py

# multi-gpu train
./tools/dist_train.sh configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-inception.pth

# single-gpu test
python tools/test.py configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-inception.pth

# multi-gpu test
./tools/dist_test.sh configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-inception.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>

## 引用

```bibtex
@InProceedings{Kupyn_2019_ICCV,
author = {Orest Kupyn and Tetiana Martyniuk and Junru Wu and Zhangyang Wang},
title = {DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```
