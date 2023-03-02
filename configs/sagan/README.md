# SAGAN (ICML'2019)

> [Self-attention generative adversarial networks](https://proceedings.mlr.press/v97/zhang19d.html)

> **Task**: Conditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

In this paper, we propose the Self-Attention Generative Adversarial Network (SAGAN) which allows attention-driven, long-range dependency modeling for image generation tasks. Traditional convolutional GANs generate high-resolution details as a function of only spatially local points in lower-resolution feature maps. In SAGAN, details can be generated using cues from all feature locations. Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other. Furthermore, recent work has shown that generator conditioning affects GAN performance. Leveraging this insight, we apply spectral normalization to the GAN generator and find that this improves training dynamics. The proposed SAGAN performs better than prior work, boosting the best published Inception score from 36.8 to 52.52 and reducing Fréchet Inception distance from 27.62 to 18.65 on the challenging ImageNet dataset. Visualization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143054130-8cc1d9b7-df13-4fdb-8dbf-af4b2c15ed28.JPG"/>
</div>

## Results and models

<div align="center">
  <b> Results from our SAGAN trained in CIFAR10</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/28132635/127619657-67f2e62d-52e4-43d2-931f-6d0e6e019813.png" width="400"/>
</div>

|                       Model                        | Dataset  | Inplace ReLU | dist_step | Total Batchsize (BZ_PER_GPU * NGPU) | Total Iters\* |  Iter  |   IS    |   FID   |                       Download                        |
| :------------------------------------------------: | :------: | :----------: | :-------: | :---------------------------------: | :-----------: | :----: | :-----: | :-----: | :---------------------------------------------------: |
| [SAGAN-32x32-woInplaceReLU Best IS](./sagan_woReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py) | CIFAR10  |     w/o      |     5     |                64x1                 |    500000     | 400000 | 9.3217  | 10.5030 | [model](https://download.openmmlab.com/mmediting/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_is-iter400000_20210730_125743-4008a9ca.pth) \| [Log](https://download.openmmlab.com/mmediting/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_20210730_125449_fid-d50568a4_is-04008a9ca.json) |
| [SAGAN-32x32-woInplaceReLU Best FID](./sagan_woReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py) | CIFAR10  |     w/o      |     5     |                64x1                 |    500000     | 480000 | 9.3174  | 9.4252  | [model](https://download.openmmlab.com/mmediting/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_fid-iter480000_20210730_125449-d50568a4.pth) \| [Log](https://download.openmmlab.com/mmediting/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_woReUinplace_20210730_125449_fid-d50568a4_is-04008a9ca.json) |
| [SAGAN-32x32-wInplaceReLU Best IS](./sagan_wReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py) | CIFAR10  |      w       |     5     |                64x1                 |    500000     | 380000 | 9.2286  | 11.7760 | [model](https://download.openmmlab.com/mmediting/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_is-iter380000_20210730_124937-c77b4d25.pth) \| [Log](https://download.openmmlab.com/mmediting/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_20210730_125155_fid-cbefb354_is-c77b4d25.json) |
| [SAGAN-32x32-wInplaceReLU Best FID](./sagan_wReLUinplace_lr2e-4-ndisc5-1xb64_cifar10-32x32.py) | CIFAR10  |      w       |     5     |                64x1                 |    500000     | 460000 | 9.2061  | 10.7781 | [model](https://download.openmmlab.com/mmediting/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_fid-iter460000_20210730_125155-cbefb354.pth) \| [Log](https://download.openmmlab.com/mmediting/sagan/sagan_cifar10_32_lr2e-4_ndisc5_b64x1_wReLUinplace_20210730_125155_fid-cbefb354_is-c77b4d25.json) |
| [SAGAN-128x128-woInplaceReLU Best IS](./sagan_woReLUinplace_Glr1e-4_Dlr4e-4_ndisc1-4xb64_imagenet1k-128x128.py) | ImageNet |     w/o      |     1     |                64x4                 |    1000000    | 980000 | 31.5938 | 36.7712 | [model](https://download.openmmlab.com/mmediting/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_is-iter980000_20210730_163140-cfbebfc6.pth) \| [Log](https://download.openmmlab.com/mmediting/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_20210730_163431_fid-d7916963_is-cfbebfc6.json) |
| [SAGAN-128x128-woInplaceReLU Best FID](./sagan_woReLUinplace_Glr1e-4_Dlr4e-4_ndisc1-4xb64_imagenet1k-128x128.py) | ImageNet |     w/o      |     1     |                64x4                 |    1000000    | 950000 | 28.4936 | 34.7838 | [model](https://download.openmmlab.com/mmediting/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_fid-iter950000_20210730_163431-d7916963.pth) \| [Log](https://download.openmmlab.com/mmediting/sagan/sagan_imagenet1k_128_Glr1e-4_Dlr4e-4_ndisc1_b32x4_woReLUinplace_20210730_163431_fid-d7916963_is-cfbebfc6.json) |
| [SAGAN-128x128-BigGAN Schedule Best IS](./sagan_woReLUinplace-Glr1e-4_Dlr4e-4_noaug-ndisc1-8xb32-bigGAN-sch_imagenet1k-128x128.py) | ImageNet |     w/o      |     1     |                32x8                 |    1000000    | 826000 | 69.5350 | 12.8295 | [model](https://download.openmmlab.com/mmediting/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.pth) \| [Log](https://download.openmmlab.com/mmediting/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.json) |
| [SAGAN-128x128-BigGAN Schedule Best FID](./sagan_woReLUinplace-Glr1e-4_Dlr4e-4_noaug-ndisc1-8xb32-bigGAN-sch_imagenet1k-128x128.py) | ImageNet |     w/o      |     1     |                32x8                 |    1000000    | 826000 | 69.5350 | 12.8295 | [model](https://download.openmmlab.com/mmediting/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.pth) \| [Log](https://download.openmmlab.com/mmediting/sagan/sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.json) |

'\*' Iteration counting rule in our implementation is different from others. If you want to align with other codebases, you can use the following conversion formula:

```
total_iters (biggan/pytorch studio gan) = our_total_iters / dist_step
```

We also provide converted pre-train models from [Pytorch-StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN).
To be noted that, in Pytorch Studio GAN, **inplace ReLU** is used in generator and discriminator.

|           Model            | Dataset  | Inplace ReLU | n_disc | Total Iters | IS (Our Pipeline) | FID (Our Pipeline) | IS (StudioGAN) | FID (StudioGAN) |           Download            |            Original Download link            |
| :------------------------: | :------: | :----------: | :----: | :---------: | :---------------: | :----------------: | :------------: | :-------------: | :---------------------------: | :------------------------------------------: |
| [SAGAN-32x32 StudioGAN](./sagan_cvt-studioGAN_cifar10-32x32.py) | CIFAR10  |      w       |   5    |   100000    |       9.116       |      10.2011       |     8.680      |     14.009      | [model](https://download.openmmlab.com/mmediting/sagan/sagan_32_cifar10_convert-studio-rgb_20210730_153321-080da7e2.pth) | [model](https://drive.google.com/drive/folders/1FA8hcz4MB8-hgTwLuDA0ZUfr8slud5P_) |
| [SAGAN0-128x128 StudioGAN](./sagan_128_cvt_studioGAN.py) | ImageNet |      w       |   1    |   1000000   |      27.367       |      40.1162       |     29.848     |     34.726      | [model](https://download.openmmlab.com/mmediting/sagan/sagan_128_imagenet1k_convert-studio-rgb_20210730_153357-eddb0d1d.pth) | [model](https://drive.google.com/drive/folders/1ZYaqeeumDgxOPDhRR5QLeLFIpgBJ9S6B) |

- `Our Pipeline` denote results evaluated with our pipeline.
- `StudioGAN` denote results released by Pytorch-StudioGAN.

For IS metric, our implementation is different from PyTorch-Studio GAN in the following aspects:

1. We use [Tero's Inception](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for feature extraction.
2. We use bicubic interpolation with PIL backend to resize image before feed them to Inception.

For FID evaluation, we follow the pipeline of [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch/blob/98459431a5d618d644d54cd1e9fceb1e5045648d/calculate_inception_moments.py#L52), where the whole training set is adopted to extract inception statistics, and Pytorch Studio GAN uses 50000 randomly selected samples. Besides, we also use [Tero's Inception](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) for feature extraction.

You can download the preprocessed inception state by the following url: [CIFAR10](https://download.openmmlab.com/mmediting/evaluation/fid_inception_pkl/cifar10.pkl) and [ImageNet1k](https://download.openmmlab.com/mmediting/evaluation/fid_inception_pkl/imagenet.pkl).

You can use following commands to extract those inception states by yourself.

```
# For CIFAR10
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/cifar10_inception_stat.py --pklname cifar10.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train

# For ImageNet1k
python tools/utils/inception_stat.py --data-cfg configs/_base_/datasets/imagenet_128x128_inception_stat.py --pklname imagenet.pkl --no-shuffle --inception-style stylegan --num-samples -1 --subset train
```

## Citation

```latex
@inproceedings{zhang2019self,
  title={Self-attention generative adversarial networks},
  author={Zhang, Han and Goodfellow, Ian and Metaxas, Dimitris and Odena, Augustus},
  booktitle={International conference on machine learning},
  pages={7354--7363},
  year={2019},
  organization={PMLR},
  url={https://proceedings.mlr.press/v97/zhang19d.html},
}
```
