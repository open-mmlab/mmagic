# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (ICLR'2016)

> [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

> **Task**: Unconditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143050281-60808c3f-81d0-4fae-9071-f4c297116b2f.JPG"/>
</div>

## Results and models

<div align="center">
  <b> DCGAN 64x64, CelebA-Cropped</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113991928-871f9b80-9885-11eb-920e-d389c603fed8.png" width="800"/>
</div>

|                                Model                                 |    Dataset     |           SWD            | MS-SSIM |                                        Download                                         |
| :------------------------------------------------------------------: | :------------: | :----------------------: | :-----: | :-------------------------------------------------------------------------------------: |
| [DCGAN 64x64](./dcgan_Glr4e-4_Dlr1e-4_1xb128-5kiters_mnist-64x64.py) | MNIST (64x64)  |  21.16, 4.4, 8.41/11.32  | 0.1395  | [model](https://download.openmmlab.com/mmediting/dcgan/dcgan_mnist-64_b128x1_Glr4e-4_Dlr1e-4_5k_20210512_163926-207a1eaf.pth) \| [log](https://download.openmmlab.com//mmgen/dcgan/dcgan_mnist-64_b128x1_Glr4e-4_Dlr1e-4_5k_20210512_163926-207a1eaf.json) |
|     [DCGAN 64x64](./dcgan_1xb128-300kiters_celeba-cropped-64.py)     | CelebA-Cropped |  8.93,10.53,50.32/23.26  | 0.2899  | [model](https://download.openmmlab.com/mmediting/dcgan/dcgan_celeba-cropped_64_b128x1_300kiter_20210408_161607-1f8a2277.pth) \| [log](https://download.openmmlab.com/mmediting/dcgan/dcgan_celeba-cropped_64_b128x1_300kiter_20210408_161607-1f8a2277.json) |
|     [DCGAN 64x64](./dcgan_1xb128-5epoches_lsun-bedroom-64x64.py)     |  LSUN-Bedroom  | 42.79, 34.55, 98.46/58.6 | 0.2095  | [model](https://download.openmmlab.com/mmediting/dcgan/dcgan_lsun-bedroom_64_b128x1_5e_20210408_161713-117c498b.pth) \| [log](https://download.openmmlab.com/mmediting/dcgan/dcgan_lsun-bedroom_64_b128x1_5e_20210408_161713-117c498b.json) |

## Citation

```latex
@article{radford2015unsupervised,
  title={Unsupervised representation learning with deep convolutional generative adversarial networks},
  author={Radford, Alec and Metz, Luke and Chintala, Soumith},
  journal={arXiv preprint arXiv:1511.06434},
  year={2015},
  url={https://arxiv.org/abs/1511.06434},
}
```
