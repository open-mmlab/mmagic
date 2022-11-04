# NAFNet (ECCV'2022)

> [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676)

> **Task**: Image Restoration

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Although there have been significant advances in the field of image restoration recently, the system complexity of the state-of-the-art (SOTA) methods is increasing as well, which may hinder the convenient analysis and comparison of methods. In this paper, we propose a simple baseline that exceeds the SOTA methods and is computationally efficient. To further simplify the baseline, we reveal that the nonlinear activation functions, e.g. Sigmoid, ReLU, GELU, Softmax, etc. are not necessary: they could be replaced by multiplication or removed. Thus, we derive a Nonlinear Activation Free Network, namely NAFNet, from the baseline. SOTA results are achieved on various challenging benchmarks, e.g. 33.69 dB PSNR on GoPro (for image deblurring), exceeding the previous SOTA 0.38 dB with only 8.4% of its computational costs; 40.30 dB PSNR on SIDD (for image denoising), exceeding the previous SOTA 0.28 dB with less than half of its computational costs.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/43229734/199919292-81d307d9-144b-4d07-9f26-0c09d86e84a5.jpg" width="400"/>
</div >

## Results and models

Coming soon.

<!-- |                            Method                             | scale | Set5 PSNR | Set5 SSIM | Set14 PSNR | Set14 SSIM | DIV2K PSNR | DIV2K SSIM |   GPU Info   |                             Download                             |
| :-----------------------------------------------------------: | :---: | :-------: | :-------: | :--------: | :--------: | :--------: | :--------: | :----------: | :--------------------------------------------------------------: |
| [liif_edsr_norm_c64b16_g1_1000k_div2k](/configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py) |  x2   |  35.7131  |  0.9366   |  31.5579   |   0.8889   |  34.6647   |   0.9355   | 1 (TITAN Xp) | [model](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.log.json) |
|                               △                               |  x3   |  32.3805  |  0.8915   |  28.4605   |   0.8039   |  30.9808   |   0.8724   |      △       |                                △                                 |
|                               △                               |  x4   |  30.2748  |  0.8509   |  26.8415   |   0.7381   |  29.0245   |   0.8187   |      △       |                                △                                 |
|                               △                               |  x6   |  27.1187  |  0.7774   |  24.7461   |   0.6444   |  26.7770   |   0.7425   |      △       |                                △                                 |
|                               △                               |  x18  |  20.8516  |  0.5406   |  20.0096   |   0.4525   |  22.1987   |   0.5955   |      △       |                                △                                 |
|                               △                               |  x30  |  18.8467  |  0.5010   |  18.1321   |   0.3963   |  20.5050   |   0.5577   |      △       |                                △                                 |
| [liif_rdn_norm_c64b16_g1_1000k_div2k](/configs/liif/liif-rdn-norm_c64b16_1xb16-1000k_div2k.py) |  x2   |  35.7874  |  0.9366   |  31.6866   |   0.8896   |  34.7548   |   0.9356   | 1 (TITAN Xp) | [model](https://download.openmmlab.com/mmediting/restorers/liif/liif_rdn_norm_c64b16_g1_1000k_div2k_20210717-22d6fdc8.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/liif/liif_rdn_norm_c64b16_g1_1000k_div2k_20210717-22d6fdc8.log.json) |
|                               △                               |  x3   |  32.4992  |  0.8923   |  28.4905   |   0.8037   |  31.0744   |   0.8731   |      △       |                                △                                 |
|                               △                               |  x4   |  30.3835  |  0.8513   |  26.8734   |   0.7373   |  29.1101   |   0.8197   |      △       |                                △                                 |
|                               △                               |  x6   |  27.1914  |  0.7751   |  24.7824   |   0.6434   |  26.8693   |   0.7437   |      △       |                                △                                 |
|                               △                               |  x18  |  20.8913  |  0.5329   |  20.1077   |   0.4537   |  22.2972   |   0.5950   |      △       |                                △                                 |
|                               △                               |  x30  |  18.9354  |  0.4864   |  18.1448   |   0.3942   |  20.5663   |   0.5560   |      △       |                                △                                 |

Note:

- △ refers to ditto.
- Evaluated on RGB channels,  `scale` pixels in each border are cropped before evaluation. -->

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py

# single-gpu train
python tools/train.py configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py

# multi-gpu train
./tools/dist_train.sh configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py /path/to/checkpoint

# single-gpu test
python tools/test.py configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py /path/to/checkpoint

# multi-gpu test
./tools/dist_test.sh configs/nafnet/nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py /path/to/checkpoint 8
```

Pretrained checkpoints will come soon.

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>

## Citation

```bibtex
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
```
