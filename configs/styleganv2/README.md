# StyleGANv2 (CVPR'2020)

> [Analyzing and Improving the Image Quality of Stylegan](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html)

> **Task**: Unconditional GANs

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven unconditional generative image modeling. We expose and analyze several of its characteristic artifacts, and propose changes in both model architecture and training methods to address them. In particular, we redesign the generator normalization, revisit progressive growing, and regularize the generator to encourage good conditioning in the mapping from latent codes to images. In addition to improving image quality, this path length regularizer yields the additional benefit that the generator becomes significantly easier to invert. This makes it possible to reliably attribute a generated image to a particular network. We furthermore visualize how well the generator utilizes its output resolution, and identify a capacity problem, motivating us to train larger models for additional quality improvements. Overall, our improved model redefines the state of the art in unconditional image modeling, both in terms of existing distribution quality metrics as well as perceived image quality.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143055738-58b15493-ab87-436d-94fd-c4c19e4a0225.JPG"/>
</div>

## Results and Models

<div align="center">
  <b> Results (compressed) from StyleGAN2 config-f trained by mmagic</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113825919-25433100-97b4-11eb-84f7-5c66b3cfbc68.png" width="800"/>
</div>

|                                  Model                                   |   Dataset   |     Comment     | FID50k | Precision50k | Recall50k |                                  Download                                   |
| :----------------------------------------------------------------------: | :---------: | :-------------: | :----: | :----------: | :-------: | :-------------------------------------------------------------------------: |
| [stylegan2_c2_8xb4_ffhq-1024x1024](./stylegan2_c2_8xb4_ffhq-1024x1024.py) |    FFHQ     | official weight | 2.8134 |    62.856    |  49.400   | [model](https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth) |
| [stylegan2_c2_8xb4_lsun-car-384x512](./stylegan2_c2_8xb4_lsun-car-384x512.py) |  LSUN_CAR   | official weight | 5.4316 |    65.986    |  48.190   | [model](https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth) |
| [stylegan2_c2_8xb4-800kiters_lsun-horse-256x256](./stylegan2_c2_8xb4-800kiters_lsun-horse-256x256.py) | LSUN_HORSE  | official weight |   -    |      -       |     -     | [model](https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth) |
| [stylegan2_c2_8xb4-800kiters_lsun-church-256x256](./stylegan2_c2_8xb4-800kiters_lsun-church-256x256.py) | LSUN_CHURCH | official weight |   -    |      -       |     -     | [model](https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth) |
| [stylegan2_c2_8xb4-800kiters_lsun-cat-256x256](./stylegan2_c2_8xb4-800kiters_lsun-cat-256x256.py) |  LSUN_CAT   | official weight |   -    |      -       |     -     | [model](https://download.openmmlab.com/mmediting/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth) |
| [stylegan2_c2_8xb4-800kiters_ffhq-256x256](./stylegan2_c2_8xb4-800kiters_ffhq-256x256.py) |    FFHQ     |  our training   | 3.992  |    69.012    |  40.417   | [model](https://download.openmmlab.com/mmediting/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth) |
| [stylegan2_c2_8xb4_ffhq-1024x1024](./stylegan2_c2_8xb4_ffhq-1024x1024.py) |    FFHQ     |  our training   | 2.8185 |    68.236    |  49.583   | [model](https://download.openmmlab.com/mmediting/stylegan2/stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth) |
| [stylegan2_c2_8xb4_lsun-car-384x512](./stylegan2_c2_8xb4_lsun-car-384x512.py) |  LSUN_CAR   |  our training   | 2.4116 |    66.760    |  50.576   | [model](https://download.openmmlab.com/mmediting/stylegan2/stylegan2_c2_lsun-car_384x512_b4x8_1800k_20210424_160929-fc9072ca.pth) |

## FP16 Support and Experiments

Currently, we have supported FP16 training for StyleGAN2, and here are the results for the mixed-precision training. (Experiments for FFHQ1024 will come soon.)

<div align="center">
  <b> Evaluation FID for FP32 and FP16 training </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/117523645-18e90880-afec-11eb-9327-da14362b8bfd.png" width="600"/>
</div>

As shown in the figure, we provide **3** ways to do mixed-precision training for `StyleGAN2`:

- [stylegan2_c2_fp16_PL-no-scaler](./stylegan2_c2-PL_8xb4-fp16-partial-GD-no-scaler-800kiters_ffhq-256x256.py): In this setting, we try our best to follow the official FP16 implementation in [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada). Similar to the official version, we only adopt FP16 training for the higher-resolution feature maps (the last 4 stages in G and the first 4 stages). Note that we do not adopt the `clamp` way to avoid gradient overflow used in the official implementation. We use the `autocast` function from `torch.cuda.amp` package.
- [stylegan2_c2_fp16-globalG-partialD_PL-R1-no-scaler](./stylegan2_c2-PL-R1_8xb4-fp16-globalG-partialD-no-scaler-800kiters_ffhq-256x256.py): In this config, we try to adopt mixed-precision training for the whole generator, but in partial discriminator (the first 4 higher-resolution stages). Note that we do not apply the loss scaler in the path length loss and gradient penalty loss. Because we always meet divergence after adopting the loss scaler to scale the gradient in these two losses.
- [stylegan2_c2_apex_fp16_PL-R1-no-scaler](./stylegan2_c2-PL-R1_8xb4-apex-fp16-no-scaler-800kiters_ffhq-256x256.py): In this setting, we adopt the [APEX](https://github.com/NVIDIA/apex) toolkit to implement mixed-precision training with multiple loss/gradient scalers. In APEX, you can assign different loss scalers for the generator and the discriminator respectively. Note that we still ignore the gradient scaler in the path length loss and gradient penalty loss.

|                                  Model                                   |                 Comment                 | Dataset | FID50k |                                   Download                                   |
| :----------------------------------------------------------------------: | :-------------------------------------: | :-----: | :----: | :--------------------------------------------------------------------------: |
| [stylegan2_c2_8xb4-800kiters_ffhq-256x256](./stylegan2_c2_8xb4-800kiters_ffhq-256x256.py) |                baseline                 | FFHQ256 | 3.992  | [ckpt](https://download.openmmlab.com/mmediting/stylegan2/stylegan2_c2_ffhq_256_b4x8_20210407_160709-7890ae1f.pth) |
| [stylegan2_c2-PL_8xb4-fp16-partial-GD-no-scaler-800kiters_ffhq-256x256](./stylegan2_c2-PL_8xb4-fp16-partial-GD-no-scaler-800kiters_ffhq-256x256.py) |         partial layers in fp16          | FFHQ256 | 4.331  | [ckpt](https://download.openmmlab.com/mmediting/stylegan2/stylegan2_c2_fp16_partial-GD_PL-no-scaler_ffhq_256_b4x8_800k_20210508_114854-dacbe4c9.pth) |
| [stylegan2_c2-PL-R1_8xb4-fp16-globalG-partialD-no-scaler-800kiters_ffhq-256x256](./stylegan2_c2-PL-R1_8xb4-fp16-globalG-partialD-no-scaler-800kiters_ffhq-256x256.py) |           the whole G in fp16           | FFHQ256 | 4.362  | [ckpt](https://download.openmmlab.com/mmediting/stylegan2/stylegan2_c2_fp16-globalG-partialD_PL-R1-no-scaler_ffhq_256_b4x8_800k_20210508_114930-ef8270d4.pth) |
| [stylegan2_c2-PL-R1_8xb4-apex-fp16-no-scaler-800kiters_ffhq-256x256](./stylegan2_c2-PL-R1_8xb4-apex-fp16-no-scaler-800kiters_ffhq-256x256.py) | the whole G&D in fp16 + two loss scaler | FFHQ256 | 4.614  | [ckpt](https://download.openmmlab.com/mmediting/stylegan2/stylegan2_c2_apex_fp16_PL-R1-no-scaler_ffhq_256_b4x8_800k_20210508_114701-c2bb8afd.pth) |

As shown in this table, `P&R50k_full` is the metric used in StyleGANv1 and StyleGANv2. `full` indicates that we use the whole dataset for extracting the real distribution, e.g., 70000 images in FFHQ dataset. However, adopting the VGG16 provided from [Tero](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt) requires that your PyTorch version must fulfill `>=1.6.0`. Be careful about using the PyTorch's VGG16 to extract features, which will cause higher precision and recall.

## Citation

```latex
@inproceedings{karras2020analyzing,
  title={Analyzing and improving the image quality of stylegan},
  author={Karras, Tero and Laine, Samuli and Aittala, Miika and Hellsten, Janne and Lehtinen, Jaakko and Aila, Timo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8110--8119},
  year={2020},
  url={https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html},
}
```
