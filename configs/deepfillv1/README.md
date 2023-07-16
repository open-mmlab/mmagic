# DeepFillv1 (CVPR'2018)

> [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892)

> **Task**: Inpainting

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Recent deep learning based approaches have shown promising results for the challenging task of inpainting large missing regions in an image. These methods can generate visually plausible image structures and textures, but often create distorted structures or blurry textures inconsistent with surrounding areas. This is mainly due to ineffectiveness of convolutional neural networks in explicitly borrowing or copying information from distant spatial locations. On the other hand, traditional texture and patch synthesis approaches are particularly suitable when it needs to borrow textures from the surrounding regions. Motivated by these observations, we propose a new deep generative model-based approach which can not only synthesize novel image structures but also explicitly utilize surrounding image features as references during network training to make better predictions. The model is a feed-forward, fully convolutional neural network which can process images with multiple holes at arbitrary locations and with variable sizes during the test time. Experiments on multiple datasets including faces (CelebA, CelebA-HQ), textures (DTD) and natural images (ImageNet, Places2) demonstrate that our proposed approach generates higher-quality inpainting results than existing ones.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144174665-9675931f-e448-4475-a659-99b65e7d4a64.png" width="400"/>
</div >

## Results and models

**CelebA-HQ**

|                       Model                       |  Mask Type  | Resolution | Train Iters |  Dataset   | l1 error |  PSNR  | SSIM  | Training Resources |                                 Download                                 |
| :-----------------------------------------------: | :---------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :----------------: | :----------------------------------------------------------------------: |
| [DeepFillv1](./deepfillv1_4xb4_celeba-256x256.py) | square bbox |  256x256   |    1500k    | CelebA-val |  6.677   | 26.878 | 0.911 |         4          | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.log.json) |

**Places365-Challenge**

|                       Model                       |  Mask Type  | Resolution | Train Iters |    Dataset    | l1 error |  PSNR  | SSIM  | Training Resources |                               Download                                |
| :-----------------------------------------------: | :---------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :----------------: | :-------------------------------------------------------------------: |
| [DeepFillv1](./deepfillv1_8xb2_places-256x256.py) | square bbox |  256x256   |    3500k    | Places365-val |  11.019  | 23.429 | 0.862 |         8          | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/deepfillv1/deepfillv1_8xb2_places-256x256.py

# single-gpu train
python tools/train.py configs/deepfillv1/deepfillv1_8xb2_places-256x256.py

# multi-gpu train
./tools/dist_train.sh configs/deepfillv1/deepfillv1_8xb2_places-256x256.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/deepfillv1/deepfillv1_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth

# single-gpu test
python tools/test.py configs/deepfillv1/deepfillv1_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth

# multi-gpu test
./tools/dist_test.sh configs/deepfillv1/deepfillv1_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@inproceedings{yu2018generative,
  title={Generative image inpainting with contextual attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5505--5514},
  year={2018}
}
```
