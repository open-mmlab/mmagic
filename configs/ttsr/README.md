# TTSR (CVPR'2020)

> [Learning Texture Transformer Network for Image Super-Resolution](https://arxiv.org/abs/2006.04139)

> **Task**: Image Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We study on image super-resolution (SR), which aims to recover realistic textures from a low-resolution (LR) image. Recent progress has been made by taking high-resolution images as references (Ref), so that relevant textures can be transferred to LR images. However, existing SR approaches neglect to use attention mechanisms to transfer high-resolution (HR) textures from Ref images, which limits these approaches in challenging cases. In this paper, we propose a novel Texture Transformer Network for Image Super-Resolution (TTSR), in which the LR and Ref images are formulated as queries and keys in a transformer, respectively. TTSR consists of four closely-related modules optimized for image generation tasks, including a learnable texture extractor by DNN, a relevance embedding module, a hard-attention module for texture transfer, and a soft-attention module for texture synthesis. Such a design encourages joint feature learning across LR and Ref images, in which deep feature correspondences can be discovered by attention, and thus accurate texture features can be transferred. The proposed texture transformer can be further stacked in a cross-scale way, which enables texture recovery from different levels (e.g., from 1x to 4x magnification). Extensive experiments show that TTSR achieves significant improvements over state-of-the-art approaches on both quantitative and qualitative evaluations.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035689-e5afa799-f469-40a0-aa94-0b84a46726a1.png" width="400"/>
</div >

## Results and models

Evaluated on CUFED dataset (RGB channels), `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR and SSIM` .

|                                   Model                                    | Dataset | scale |  PSNR   |  SSIM  | Training Resources |                                      Download                                       |
| :------------------------------------------------------------------------: | :-----: | :---: | :-----: | :----: | :----------------: | :---------------------------------------------------------------------------------: |
| [ttsr-rec_x4_c64b16_g1_200k_CUFED](./ttsr-rec_x4c64b16_1xb9-200k_CUFED.py) |  CUFED  |  x4   | 25.2433 | 0.7491 |    1 (TITAN Xp)    | [model](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-rec_x4_c64b16_g1_200k_CUFED_20210525-b0dba584.log.json) |
| [ttsr-gan_x4_c64b16_g1_500k_CUFED](./ttsr-gan_x4c64b16_1xb9-500k_CUFED.py) |  CUFED  |  x4   | 24.6075 | 0.7234 |    1 (TITAN Xp)    | [model](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py

# single-gpu train
python tools/train.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py

# multi-gpu train
./tools/dist_train.sh configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth

# single-gpu test
python tools/test.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth

# multi-gpu test
./tools/dist_test.sh configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py https://download.openmmlab.com/mmediting/restorers/ttsr/ttsr-gan_x4_c64b16_g1_500k_CUFED_20210626-2ab28ca0.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@inproceedings{yang2020learning,
  title={Learning texture transformer network for image super-resolution},
  author={Yang, Fuzhi and Yang, Huan and Fu, Jianlong and Lu, Hongtao and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5791--5800},
  year={2020}
}
```
