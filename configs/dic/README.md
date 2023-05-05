# DIC (CVPR'2020)

> [Deep Face Super-Resolution with Iterative Collaboration between Attentive Recovery and Landmark Estimation](https://arxiv.org/abs/2003.13063)

> **Task**: Image Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Recent works based on deep learning and facial priors have succeeded in super-resolving severely degraded facial images. However, the prior knowledge is not fully exploited in existing methods, since facial priors such as landmark and component maps are always estimated by low-resolution or coarsely super-resolved images, which may be inaccurate and thus affect the recovery performance. In this paper, we propose a deep face super-resolution (FSR) method with iterative collaboration between two recurrent networks which focus on facial image recovery and landmark estimation respectively. In each recurrent step, the recovery branch utilizes the prior knowledge of landmarks to yield higher-quality images which facilitate more accurate landmark estimation in turn. Therefore, the iterative information interaction between two processes boosts the performance of each other progressively. Moreover, a new attentive fusion module is designed to strengthen the guidance of landmark maps, where facial components are generated individually and aggregated attentively for better restoration. Quantitative and qualitative experimental results show the proposed method significantly outperforms state-of-the-art FSR methods in recovering high-quality face images.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144017838-63e31123-1b59-4743-86bb-737bd32a9209.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

In the log data of `dic_gan_x8c48b6_g4_150k_CelebAHQ`, DICGAN is verified on the first 9 pictures of the test set of CelebA-HQ, so `PSNR` and `SSIM` shown in the follow table is different from the log data.

`Training Resources`: Training Resourcesrmation during training.

|                                    Model                                     | Dataset  | scale |  PSNR   |  SSIM  | Training Resources  |                                    Download                                     |
| :--------------------------------------------------------------------------: | :------: | :---: | :-----: | :----: | :-----------------: | :-----------------------------------------------------------------------------: |
|     [dic_x8c48b6_g4_150k_CelebAHQ](./dic_x8c48b6_4xb2-150k_celeba-hq.py)     | CelebAHQ |  x8   | 25.2319 | 0.7422 | 4 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.log.json) |
| [dic_gan_x8c48b6_g4_500k_CelebAHQ](./dic_gan-x8c48b6_4xb2-500k_celeba-hq.py) | CelebAHQ |  x8   | 23.6241 | 0.6721 | 4 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py

# single-gpu train
python tools/train.py configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py

# multi-gpu train
./tools/dist_train.sh configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth

# single-gpu test
python tools/test.py configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth

# multi-gpu test
./tools/dist_test.sh configs/dic/dic_gan-x8c48b6_4xb2-500k_celeba-hq.py https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@inproceedings{ma2020deep,
  title={Deep face super-resolution with iterative collaboration between attentive recovery and landmark estimation},
  author={Ma, Cheng and Jiang, Zhenyu and Rao, Yongming and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5569--5578},
  year={2020}
}
```
