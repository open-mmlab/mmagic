# Global&Local (ToG'2017)

> [Globally and Locally Consistent Image Completion](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)

> **Task**: Inpainting

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present a novel approach for image completion that results in images that are both locally and globally consistent. With a fully-convolutional neural network, we can complete images of arbitrary resolutions by flling in missing regions of any shape. To train this image completion network to be consistent, we use global and local context discriminators that are trained to distinguish real images from completed ones. The global discriminator looks at the entire image to assess if it is coherent as a whole, while the local discriminator looks only at a small area centered at the completed region to ensure the local consistency of the generated patches. The image completion network is then trained to fool the both context discriminator networks, which requires it to generate images that are indistinguishable from real ones with regard to overall consistency as well as in details. We show that our approach can be used to complete a wide variety of scenes. Furthermore, in contrast with the patch-based approaches such as PatchMatch, our approach can generate fragments that do not appear elsewhere in the image, which allows us to naturally complete the image.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175196-51dfda11-f7e1-4c7e-abed-42799f757bef.png" width="400"/>
</div >

## Results and models

*Note that we do not apply the post-processing module in Global&Local for a fair comparison with current deep inpainting methods.*

|                    Model                     |       Dataset       |  Mask Type  | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  | Training Resources |                        Download                         |
| :------------------------------------------: | :-----------------: | :---------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :----------------: | :-----------------------------------------------------: |
| [Global&Local](./gl_8xb12_places-256x256.py) | Places365-Challenge | square bbox |  256x256   |    500k     | Places365-val |  11.164  | 23.152 | 0.862 |         8          | [model](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.log.json) |
| [Global&Local](./gl_8xb12_celeba-256x256.py) |      CelebA-HQ      | square bbox |  256x256   |    500k     |  CelebA-val   |  6.678   | 26.780 | 0.904 |         8          | [model](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/global_local/gl_8xb12_places-256x256.py

# single-gpu train
python tools/train.py configs/global_local/gl_8xb12_places-256x256.py

# multi-gpu train
./tools/dist_train.sh configs/global_local/gl_8xb12_places-256x256.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/global_local/gl_8xb12_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.pth

# single-gpu test
python tools/test.py configs/global_local/gl_8xb12_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.pth

# multi-gpu test
./tools/dist_test.sh configs/global_local/gl_8xb12_places-256x256.py https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@article{iizuka2017globally,
  title={Globally and locally consistent image completion},
  author={Iizuka, Satoshi and Simo-Serra, Edgar and Ishikawa, Hiroshi},
  journal={ACM Transactions on Graphics (ToG)},
  volume={36},
  number={4},
  pages={1--14},
  year={2017},
  publisher={ACM New York, NY, USA}
}
```
