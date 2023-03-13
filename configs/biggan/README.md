# BigGAN (ICLR'2019)

> [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://openreview.net/forum?id=B1xsqj09Fm)

> **Task**: Conditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Despite recent progress in generative image modeling, successfully generating high-resolution, diverse samples from complex datasets such as ImageNet remains an elusive goal. To this end, we train Generative Adversarial Networks at the largest scale yet attempted, and study the instabilities specific to such scale. We find that applying orthogonal regularization to the generator renders it amenable to a simple "truncation trick," allowing fine control over the trade-off between sample fidelity and variety by reducing the variance of the Generator's input. Our modifications lead to models which set the new state of the art in class-conditional image synthesis. When trained on ImageNet at 128x128 resolution, our models (BigGANs) achieve an Inception Score (IS) of 166.5 and Frechet Inception Distance (FID) of 7.4, improving over the previous best IS of 52.52 and FID of 18.6.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143154280-4cb22e16-92c8-4b34-9e2c-6357ed0bdac8.png"/>
</div>

## Introduction

The `BigGAN/BigGAN-Deep` is a conditional generation model that can generate both high-resolution and high-quality images by scaling up the batch size and the number of model parameters.

We have finished training `BigGAN` in `Cifar10` (32x32) and are aligning training performance in `ImageNet1k` (128x128). Some sampled results are shown below for your reference.

<div align="center">
  <b> Results from our BigGAN trained in CIFAR10</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126476913-3ce8e2c8-f189-4caa-90ed-b44e279cb669.png" width="800"/>
</div>

<div align="center">
  <b> Results from our BigGAN trained in ImageNet</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/127615534-6278ce1b-5cff-4189-83c6-9ecc8de08dfc.png" width="800"/>
</div>

Evaluation of our trained BigGAN.

|                                    Model                                    |  Dataset   |    FID (Iter)     |      IS (Iter)      |                                    Download                                     |
| :-------------------------------------------------------------------------: | :--------: | :---------------: | :-----------------: | :-----------------------------------------------------------------------------: |
|          [BigGAN 32x32](./biggan_2xb25-500kiters_cifar10-32x32.py)          |  CIFAR10   |   9.78(390000)    |    8.70(390000)     | [model](https://download.openmmlab.com/mmediting/biggan/biggan_cifar10_32x32_b25x2_500k_20210728_110906-08b61a44.pth)\|[log](https://download.openmmlab.com/mmediting/biggan/biggan_cifar10_32_b25x2_500k_20210706_171051.log.json) |
| [BigGAN 128x128 Best FID](./biggan_ajbrock-sn_8xb32-1500kiters_imagenet1k-128x128.py) | ImageNet1k | **8.69**(1232000) |   101.15(1232000)   | [model](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_b32x8_best_fid_iter_1232000_20211111_122548-5315b13d.pth)\|[log](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_b32x8_1500k_20211111_122548-5315b13d.log.json) |
| [BigGAN 128x128 Best IS](./biggan_ajbrock-sn_8xb32-1500kiters_imagenet1k-128x128.py) | ImageNet1k |  13.51(1328000)   | **129.07**(1328000) | [model](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_b32x8_best_is_iter_1328000_20211111_122911-28c688bc.pth)\|[log](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_b32x8_1500k_20211111_122548-5315b13d.log.json) |

### Note on reproducibility

`BigGAN 128x128` model is trained with V100 GPUs and CUDA 10.1 and can hardly reproduce the result with A100 and CUDA 11.3. If you have any idea about the reproducibility, please feel free to contact with us.

## Converted weights

Since we haven't finished training our models, we provide you with several pre-trained weights which have been evaluated. Here, we refer to [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) and [pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN).

Evaluation results and download links are provided below.

|                        Model                         |  Dataset   |   FID   |   IS    |                        Download                         |                        Original Download link                         |
| :--------------------------------------------------: | :--------: | :-----: | :-----: | :-----------------------------------------------------: | :-------------------------------------------------------------------: |
| [BigGAN 128x128](./biggan_cvt-BigGAN-PyTorch-rgb_imagenet1k-128x128.py) | ImageNet1k | 10.1414 | 96.728  | [model](https://download.openmmlab.com/mmediting/biggan/biggan_imagenet1k_128x128_cvt_BigGAN-PyTorch_rgb_20210730_125223-3e353fef.pth) | [link](https://drive.google.com/open?id=1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW) |
| [BigGAN-Deep 128x128](./biggan-deep_cvt-hugging-face-rgb_imagenet1k-128x128.py) | ImageNet1k | 5.9471  | 107.161 | [model](https://download.openmmlab.com/mmediting/biggan/biggan-deep_imagenet1k_128x128_cvt_hugging-face_rgb_20210728_111659-099e96f9.pth) | [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin) |
| [BigGAN-Deep 256x256](./biggan-deep_cvt-hugging-face_rgb_imagenet1k-256x256.py) | ImageNet1k | 11.3151 | 135.107 | [model](https://download.openmmlab.com/mmediting/biggan/biggan-deep_imagenet1k_256x256_cvt_hugging-face_rgb_20210728_111735-28651569.pth) | [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin) |
| [BigGAN-Deep 512x512](./biggan-deep_cvt-hugging-face_rgb_imagenet1k-512x512.py) | ImageNet1k | 16.8728 | 124.368 | [model](https://download.openmmlab.com/mmediting/biggan/biggan-deep_imagenet1k_512x512_cvt_hugging-face_rgb_20210728_112346-a42585f2.pth) | [link](https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin) |

Sampling results are shown below.

<div align="center">
  <b> Results from our BigGAN-Deep with Pre-trained weights in ImageNet 128x128 with truncation factor 0.4</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126481730-8da7180b-7b1b-42f0-9bec-78d879b6265b.png" width="800"/>
</div>

<div align="center">
  <b> Results from our BigGAN-Deep with Pre-trained weights in ImageNet 256x256 with truncation factor 0.4</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126486040-64effa29-959e-4e43-bcae-15925a2e0599.png" width="800"/>
</div>

<div align="center">
  <b> Results from our BigGAN-Deep with Pre-trained weights in ImageNet 512x512 truncation factor 0.4</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126487428-50101454-59cb-469d-a1f1-36ffb6291582.png" width="800"/>
</div>
Sampling with truncation trick above can be performed by command below.

```bash
python demo/conditional_demo.py CONFIG_PATH CKPT_PATH --sample-cfg truncation=0.4 # set truncation value as you want
```

For converted weights, we provide model configs under `configs/_base_/models` listed as follows:

```bash
# biggan_cvt-BigGAN-PyTorch-rgb_imagenet1k-128x128.py
# biggan-deep_cvt-hugging-face-rgb_imagenet1k-128x128.py
# biggan-deep_cvt-hugging-face_rgb_imagenet1k-256x256.py
# biggan-deep_cvt-hugging-face_rgb_imagenet1k-512x512.py
```

## Interpolation

To perform image Interpolation on BigGAN(or other conditional models), run

```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH
```

<div align="center">
  <b> Image interpolating Results of our BigGAN-Deep</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/126580403-2baa987b-ff55-4fb5-a53a-b08e8a6a72a2.png" width="800"/>
</div>

To perform image Interpolation on BigGAN with fixed noise, run

```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH --fix-z
```

<div align="center">
  <b> Image interpolating Results of our BigGAN-Deep with fixed noise</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/128123804-6df1dfca-1057-4b96-8428-787a86f81ef1.png" width="800"/>
</div>
To perform image Interpolation on BigGAN with fixed label, run

```bash
python apps/conditional_interpolate.py CONFIG_PATH  CKPT_PATH  --samples-path SAMPLES_PATH --fix-y
```

<div align="center">
  <b> Image interpolating Results of our BigGAN-Deep with fixed label</b>
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
