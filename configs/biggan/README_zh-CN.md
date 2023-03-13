# BigGAN (ICLR'2019)

> [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://openreview.net/forum?id=B1xsqj09Fm)

> **任务**: 条件生成对抗网络

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Despite recent progress in generative image modeling, successfully generating high-resolution, diverse samples from complex datasets such as ImageNet remains an elusive goal. To this end, we train Generative Adversarial Networks at the largest scale yet attempted, and study the instabilities specific to such scale. We find that applying orthogonal regularization to the generator renders it amenable to a simple "truncation trick," allowing fine control over the trade-off between sample fidelity and variety by reducing the variance of the Generator's input. Our modifications lead to models which set the new state of the art in class-conditional image synthesis. When trained on ImageNet at 128x128 resolution, our models (BigGANs) achieve an Inception Score (IS) of 166.5 and Frechet Inception Distance (FID) of 7.4, improving over the previous best IS of 52.52 and FID of 18.6.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143154280-4cb22e16-92c8-4b34-9e2c-6357ed0bdac8.png"/>
</div>

## Introduction

`BigGAN/BigGAN-Deep`是一个条件生成模型，通过扩大批次大小和模型参数的数量，可以生成高分辨率和高质量的图像。

我们已经在`Cifar10`（32x32）中完成了`BigGAN`的训练，并在`ImageNet1k`（128x128）上对齐了训练性能。下面是一些抽样的结果，供你参考。

<div align="center">
  <b> 我们在 CIFAR10 上训练的 BigGAN 的结果</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126476913-3ce8e2c8-f189-4caa-90ed-b44e279cb669.png" width="800"/>
</div>

<div align="center">
  <b> 我们在 ImageNet 上训练的 BigGAN 的结果</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/127615534-6278ce1b-5cff-4189-83c6-9ecc8de08dfc.png" width="800"/>
</div>

对我们训练的 BigGAN 进行评估.

|                                     算法                                      |   数据集   |    FID (Iter)     |      IS (Iter)      |                                     下载                                      |
| :---------------------------------------------------------------------------: | :--------: | :---------------: | :-----------------: | :---------------------------------------------------------------------------: |
|           [BigGAN 32x32](./biggan_2xb25-500kiters_cifar10-32x32.py)           |  CIFAR10   |   9.78(390000)    |    8.70(390000)     | [model](https://download.openmmlab.com/mmediting/biggan/biggan_cifar10_32x32_b25x2_500k_20210728_110906-08b61a44.pth)\|[log](https://download.openmmlab.com/mmediting/biggan/biggan_cifar10_32_b25x2_500k_20210706_171051.log.json) |
| [BigGAN 128x128 Best FID](./biggan_ajbrock-sn_8xb32-1500kiters_imagenet1k-128x128.py) | ImageNet1k | **8.69**(1232000) |   101.15(1232000)   | [model](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_b32x8_best_fid_iter_1232000_20211111_122548-5315b13d.pth)\|[log](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_b32x8_1500k_20211111_122548-5315b13d.log.json) |
| [BigGAN 128x128 Best IS](./biggan_ajbrock-sn_8xb32-1500kiters_imagenet1k-128x128.py) | ImageNet1k |  13.51(1328000)   | **129.07**(1328000) | [model](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_b32x8_best_is_iter_1328000_20211111_122911-28c688bc.pth)\|[log](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_b32x8_1500k_20211111_122548-5315b13d.log.json) |

### 关于可复现性的说明

`BigGAN 128x128`模型是用 V100 GPU 和 CUDA 10.1 训练的，用 A100 和 CUDA 11.3 很难再现结果。如果你对复现有任何想法，请随时与我们联系。

## 转换后的权重

由于我们还没有完成对模型的训练，我们为您提供了几个已经评估过的预训练权重。这里，我们指的是[BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)和[pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN)。

下面提供了评估结果和下载链接

|                           模型                           |   数据集   |   FID   |   IS    |                           下载                            |                        原始权重下载链接                         |
| :------------------------------------------------------: | :--------: | :-----: | :-----: | :-------------------------------------------------------: | :-------------------------------------------------------------: |
| [BigGAN 128x128](./biggan_cvt-BigGAN-PyTorch-rgb_imagenet1k-128x128.py) | ImageNet1k | 10.1414 | 96.728  | [model](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_cvt_BigGAN-PyTorch_rgb_20210730_125223-3e353fef.pth) | [link](https://drive.google.com/open?id=1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW) |
| [BigGAN-Deep 128x128](./biggan-deep_cvt-hugging-face-rgb_imagenet1k-128x128.py) | ImageNet1k | 5.9471  | 107.161 | [model](https://download.openmmlab.com/mmediting/biggan/biggan-deep_imagenet1k_128x128_cvt_hugging-face_rgb_20210728_111659-099e96f9.pth) | [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin) |
| [BigGAN-Deep 256x256](./biggan-deep_cvt-hugging-face_rgb_imagenet1k-256x256.py) | ImageNet1k | 11.3151 | 135.107 | [model](https://download.openmmlab.com/mmediting/biggan/biggan-deep_imagenet1k_256x256_cvt_hugging-face_rgb_20210728_111735-28651569.pth) | [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin) |
| [BigGAN-Deep 512x512](./biggan-deep_cvt-hugging-face_rgb_imagenet1k-512x512.py) | ImageNet1k | 16.8728 | 124.368 | [model](https://download.openmmlab.com/mmediting/biggan/biggan-deep_imagenet1k_512x512_cvt_hugging-face_rgb_20210728_112346-a42585f2.pth) | [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin) |

采样结果如下。

<div align="center">
  <b> BigGAN-Deep 在 ImageNet 128x128 中使用预训练权重的结果，截断因子为 0.4 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126481730-8da7180b-7b1b-42f0-9bec-78d879b6265b.png" width="800"/>
</div>

<div align="center">
  <b> BigGAN-Deep 在 ImageNet 256x256 中使用预训练权重的结果，截断因子为 0.4 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126486040-64effa29-959e-4e43-bcae-15925a2e0599.png" width="800"/>
</div>

<div align="center">
  <b> BigGAN-Deep 在 ImageNet 512x512 中使用预训练权重的结果，截断因子为 0.4 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126487428-50101454-59cb-469d-a1f1-36ffb6291582.png" width="800"/>
</div>
上面的截断取样技巧可以通过下面的命令进行。

```bash
python demo/conditional_demo.py CONFIG_PATH CKPT_PATH --sample-cfg truncation=0.4 # set truncation value as you want
```

对于转换后的权重，我们在`configs/_base_/models`下提供模型配置，列举如下。

```bash
# biggan_cvt-BigGAN-PyTorch-rgb_imagenet1k-128x128.py
# biggan-deep_cvt-hugging-face-rgb_imagenet1k-128x128.py
# biggan-deep_cvt-hugging-face_rgb_imagenet1k-256x256.py
# biggan-deep_cvt-hugging-face_rgb_imagenet1k-512x512.py
```

## Interpolation

要在 BigGAN（或其他条件模型）上执行图像插值，请运行

```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH
```

<div align="center">
  <b> 我们的 BigGAN-Deep 的图像插值结果</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126580403-2baa987b-ff55-4fb5-a53a-b08e8a6a72a2.png" width="800"/>
</div>

要在 BigGAN 上进行具有固定噪声的图像插值，请运行

```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH --fix-z
```

<div align="center">
  <b> 我们的 BigGAN-Deep 在固定噪音下的图像插值结果 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/128123804-6df1dfca-1057-4b96-8428-787a86f81ef1.png" width="800"/>
</div>
要在 BigGAN 上执行具有固定标签的图像插值，请运行

```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH --fix-y
```

<div align="center">
  <b> 我们的 BigGAN-Deep 带有固定标签的图像插值结果</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/128124596-421396f1-3f23-4098-b629-b00d29d710a9.png" width="800"/>
</div>

## Citation

```latex
@inproceedings{
    brock2018large,
    title={Large Scale {GAN} Training for High Fidelity Natural Image Synthesis},
    author={Andrew Brock and Jeff Donahue and Karen Simonyan},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=B1xsqj09Fm},
}
```
