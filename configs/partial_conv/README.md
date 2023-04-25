# PConv (ECCV'2018)

> [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)

> **Task**: Inpainting

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Existing deep learning based image inpainting methods use a standard convolutional network over the corrupted image, using convolutional filter responses conditioned on both valid pixels as well as the substitute values in the masked holes (typically the mean value). This often leads to artifacts such as color discrepancy and blurriness. Post-processing is usually used to reduce such artifacts, but are expensive and may fail. We propose the use of partial convolutions, where the convolution is masked and renormalized to be conditioned on only valid pixels. We further include a mechanism to automatically generate an updated mask for the next layer as part of the forward pass. Our model outperforms other methods for irregular masks. We show qualitative and quantitative comparisons with other methods to validate our approach.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175613-1bc9ad1b-072d-4c1f-a97d-1af5be2590bd.png" width="400"/>
</div >

## Results and models

|                         Model                          | Mask Type | Resolution | Train Iters |    Dataset    | l1 error |  PSNR  | SSIM  | Training Resources |                              Download                              |
| :----------------------------------------------------: | :-------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :----------------: | :----------------------------------------------------------------: |
| [PConv_Stage1](./pconv_stage1_8xb12_places-256x256.py) | free-form |  256x256   |    500k     | Places365-val |    -     |   -    |   -   |         8          |                                 -                                  |
| [PConv_Stage2](./pconv_stage2_4xb2_places-256x256.py)  | free-form |  256x256   |    500k     | Places365-val |  8.776   | 22.762 | 0.801 |         4          | [model](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.log.json) |
| [PConv_Stage1](./pconv_stage1_8xb1_celeba-256x256.py)  | free-form |  256x256   |    500k     |  CelebA-val   |    -     |   -    |   -   |         8          |                                 -                                  |
| [PConv_Stage2](./pconv_stage2_4xb2_celeba-256x256.py)  | free-form |  256x256   |    500k     |  CelebA-val   |  5.990   | 25.404 | 0.853 |         4          | [model](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/partial_conv/pconv_stage2_4xb2_places-256x256.py

# single-gpu train
python tools/train.py configs/partial_conv/pconv_stage2_4xb2_places-256x256.py

# multi-gpu train
./tools/dist_train.sh configs/partial_conv/pconv_stage2_4xb2_places-256x256.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/partial_conv/pconv_stage2_4xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth

# single-gpu test
python tools/test.py configs/partial_conv/pconv_stage2_4xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth

# multi-gpu test
./tools/dist_test.sh configs/partial_conv/pconv_stage2_4xb2_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@inproceedings{liu2018image,
  title={Image inpainting for irregular holes using partial convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={85--100},
  year={2018}
}
```
