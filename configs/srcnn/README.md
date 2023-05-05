# SRCNN (TPAMI'2015)

> [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)

> **Task**: Image Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We propose a deep learning method for single image super-resolution (SR). Our method directly learns an end-to-end mapping between the low/high-resolution images. The mapping is represented as a deep convolutional neural network (CNN) that takes the low-resolution image as the input and outputs the high-resolution one. We further show that traditional sparse-coding-based SR methods can also be viewed as a deep convolutional network. But unlike traditional methods that handle each component separately, our method jointly optimizes all layers. Our deep CNN has a lightweight structure, yet demonstrates state-of-the-art restoration quality, and achieves fast speed for practical on-line usage. We explore different network structures and parameter settings to achieve trade-offs between performance and speed. Moreover, we extend our network to cope with three color channels simultaneously, and show better overall reconstruction quality.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144034831-79f48aae-196e-42e7-92b9-069149733e3e.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                Model                                 | Dataset |  PSNR   |  SSIM  | Training Resources |                                            Download                                            |
| :------------------------------------------------------------------: | :-----: | :-----: | :----: | :----------------: | :--------------------------------------------------------------------------------------------: |
| [srcnn_x4k915_1x16_1000k_div2k](./srcnn_x4k915_1xb16-1000k_div2k.py) |  Set5   | 28.4316 | 0.8099 |         1          | [model](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608_120159.log.json) |
| [srcnn_x4k915_1x16_1000k_div2k](./srcnn_x4k915_1xb16-1000k_div2k.py) |  Set14  | 25.6486 | 0.7014 |         1          | [model](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608_120159.log.json) |
| [srcnn_x4k915_1x16_1000k_div2k](./srcnn_x4k915_1xb16-1000k_div2k.py) |  DIV2K  | 27.7460 | 0.7854 |         1          | [model](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608_120159.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py

# single-gpu train
python tools/train.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py

# multi-gpu train
./tools/dist_train.sh configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth

# single-gpu test
python tools/test.py configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth

# multi-gpu test
./tools/dist_test.sh configs/srcnn/srcnn_x4k915_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@article{dong2015image,
  title={Image super-resolution using deep convolutional networks},
  author={Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={38},
  number={2},
  pages={295--307},
  year={2015},
  publisher={IEEE}
}
```
