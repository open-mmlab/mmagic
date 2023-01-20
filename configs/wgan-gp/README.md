# WGAN-GP (NeurIPS'2017)

> [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

> **Task**: Unconditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143154792-de359728-101b-4ad1-90c0-ef3c1572d184.png"/>
</div>

## Results and models

<div align="center">
  <b> WGAN-GP 128, CelebA-Cropped</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113997469-c00e3f00-988a-11eb-81dc-19b05698b74b.png" width="800"/>
</div>

|                              Model                               |    Dataset     |      Details       |              SWD              | MS-SSIM |                               Download                               |
| :--------------------------------------------------------------: | :------------: | :----------------: | :---------------------------: | :-----: | :------------------------------------------------------------------: |
| [WGAN-GP 128](./wgangp_GN_1xb64-160kiters_celeba-cropped-128x128.py) | CelebA-Cropped |         GN         | 5.87, 9.76, 9.43, 18.84/10.97 | 0.2601  | [model](https://download.openmmlab.com/mmediting/wgangp/wgangp_GN_celeba-cropped_128_b64x1_160k_20210408_170611-f8a99336.pth) |
| [WGAN-GP 128](./wgangp_GN-GP-50_1xb64-160kiters_lsun-bedroom-128x128.py) |  LSUN-Bedroom  | GN, GP-lambda = 50 | 11.7, 7.87, 9.82, 25.36/13.69 |  0.059  | [model](https://download.openmmlab.com/mmediting/wgangp/wgangp_GN_GP-50_lsun-bedroom_128_b64x1_130k_20210408_170509-56f2a37c.pth) |

## Citation

```latex
@article{gulrajani2017improved,
  title={Improved Training of Wasserstein GANs},
  author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron},
  journal={arXiv preprint arXiv:1704.00028},
  year={2017},
  url={https://arxiv.org/abs/1704.00028},
}
```
