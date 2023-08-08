# DeblurGAN-v2 (ICCV'2019)

> [DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826)

> **Task**: Deblurring

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present a new end-to-end generative adversarial network (GAN) for single image motion deblurring, named DeblurGAN-v2, which considerably boosts state-of-the-art deblurring efficiency, quality, and flexibility. DeblurGAN-v2 is based on a relativistic conditional GAN with a double-scale discriminator. For the first time, we introduce the Feature Pyramid Network into deblurring, as a core building block in the generator of DeblurGAN-v2. It can flexibly work with a wide range of backbones, to navigate the balance between performance and efficiency. The plug-in of sophisticated backbones (e.g., Inception-ResNet-v2) can lead to solid state-of-the-art deblurring. Meanwhile, with light-weight backbones (e.g., MobileNet and its variants), DeblurGAN-v2 reaches 10-100 times faster than the nearest competitors, while maintaining close to state-of-the-art results, implying the option of real-time video deblurring. We demonstrate that DeblurGAN-v2 obtains very competitive performance on several popular benchmarks, in terms of deblurring quality (both objective and subjective), as well as efficiency. Besides, we show the architecture to be effective for general image restoration tasks too.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/master/doc_images/pipeline.jpg"/>
</div>

## Results and models

<div align="center">
  <b> DEBLURGANv2 256x256</b>
  <br/>
  <img src="https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/master/doc_images/kohler_visual.png" width="800"/>
  <img src="https://raw.githubusercontent.com/VITA-Group/DeblurGANv2/master/doc_images/restore_visual.png" width="800"/>
 </div>

|                           Model                            |      Dataset       |      G Model       |  D Model   | PSNR/<br/>SSIM |                                      Download                                      |
| :--------------------------------------------------------: | :----------------: | :----------------: | :--------: | :------------: | :--------------------------------------------------------------------------------: |
| [fpn_inception](./deblurganv2_fpn-inception_1xb1_gopro.py) | GoPro Test Dataset | InceptionResNet-v2 | double_gan |  29.55/ 0.934  | [model](https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-inception.pth) \\ [log](<>) |
| [fpn_mobilenet](./deblurganv2_fpn-mobilenet_1xb1_gopro.py) | GoPro Test Dataset |     MobileNet      | double_gan |  28.17/ 0.925  | [model](https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-mobilenet.pth) \\ [log](<>) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py

# single-gpu train
python tools/train.py configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py

# multi-gpu train
./tools/dist_train.sh configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-inception.pth

# single-gpu test
python tools/test.py configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-inception.pth

# multi-gpu test
./tools/dist_test.sh configs/deblurganv2/deblurganv2_fpn-inception_1xb1_gopro.py https://download.openxlab.org.cn/models/xiaomile/DeblurGANv2/weight/DeblurGANv2_fpn-inception.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@InProceedings{Kupyn_2019_ICCV,
author = {Orest Kupyn and Tetiana Martyniuk and Junru Wu and Zhangyang Wang},
title = {DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```
