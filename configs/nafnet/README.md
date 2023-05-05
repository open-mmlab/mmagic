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

|                                  Model                                  | Dataset | image size |       PSNR       |      SSIM      | GPU Info |                                  Download                                  |
| :---------------------------------------------------------------------: | :-----: | :--------: | :--------------: | :------------: | :------: | :------------------------------------------------------------------------: |
| [nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd](./nafnet_c64eb2248mb12db2222_8xb8-lr1e-3-400k_sidd.py) |  SIDD   |  256X256   | 40.3045(40.3045) | 0.9253(0.9614) | 1 (A100) | [model](https://download.openmmlab.com/mmediting/nafnet/NAFNet-SIDD-midc64.pth) \| log(coming soon) |
| [nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro](./nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro.py) |  GoPro  |  1280x720  | 33.7246(33.7103) | 0.9479(0.9668) | 1 (A100) | [model](https://download.openmmlab.com/mmediting/nafnet/NAFNet-GoPro-midc64.pth) \| log(coming soon) |

Note:

- a(b) where a denotes the value run by MMagic, b denotes the value copied from the original paper.
- PSNR is evaluated on RGB channels.
- SSIM is evaluated by averaging SSIMs on RGB channels, however the original paper uses the 3D SSIM kernel.

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

For more details, you can refer to **Train a model** part in [train_test.md](../../docs/en/user_guides/train_test.md).

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

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](../../docs/en/user_guides/train_test.md).

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
