# DeepFillv2 (CVPR'2019)

> [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)

> **Task**: Inpainting

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present a generative image inpainting system to complete images with free-form mask and guidance. The system is based on gated convolutions learned from millions of images without additional labelling efforts. The proposed gated convolution solves the issue of vanilla convolution that treats all input pixels as valid ones, generalizes partial convolution by providing a learnable dynamic feature selection mechanism for each channel at each spatial location across all layers. Moreover, as free-form masks may appear anywhere in images with any shape, global and local GANs designed for a single rectangular mask are not applicable. Thus, we also present a patch-based GAN loss, named SN-PatchGAN, by applying spectral-normalized discriminator on dense image patches. SN-PatchGAN is simple in formulation, fast and stable in training. Results on automatic image inpainting and user-guided extension demonstrate that our system generates higher-quality and more flexible results than previous methods. Our system helps user quickly remove distracting objects, modify image layouts, clear watermarks and edit faces.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175160-75473789-924f-490b-ab25-4c4f252fa55f.png" width="400"/>
</div >

## Results and models

**CelebA-HQ**

|                       Model                       | Mask Type | Resolution | Train Iters |  Dataset   | l1 error |  PSNR  | SSIM  | Training Resources |                                  Download                                  |
| :-----------------------------------------------: | :-------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :----------------: | :------------------------------------------------------------------------: |
| [DeepFillv2](./deepfillv2_8xb2_celeba-256x256.py) | free-form |  256x256   |     20k     | CelebA-val |  5.411   | 25.721 | 0.871 |         8          | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.log.json) |

**Places365-Challenge**

|                       Model                       | Mask Type | Resolution | Train Iters |    Dataset    | l1 error |  PSNR  | SSIM  | Training Resources |                                Download                                 |
| :-----------------------------------------------: | :-------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :----------------: | :---------------------------------------------------------------------: |
| [DeepFillv2](./deepfillv2_8xb2_places-256x256.py) | free-form |  256x256   |    100k     | Places365-val |  8.635   | 22.398 | 0.815 |         8          | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/deepfillv2/deepfillv2_8xb2_places-256x256.py

# single-gpu train
python tools/train.py configs/deepfillv2/deepfillv2_8xb2_places-256x256.py

# multi-gpu train
./tools/dist_train.sh configs/deepfillv2/deepfillv2_8xb2_places-256x256.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/deepfillv2/deepfillv2_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth

# single-gpu test
python tools/test.py configs/deepfillv2/deepfillv2_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth

# multi-gpu test
./tools/dist_test.sh configs/deepfillv2/deepfillv2_8xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@inproceedings{yu2019free,
  title={Free-form image inpainting with gated convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4471--4480},
  year={2019}
}
```
