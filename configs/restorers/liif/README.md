# Learning Continuous Image Representation with Local Implicit Image Function

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@article{chen2020learning,
  title={Learning Continuous Image Representation with Local Implicit Image Function},
  author={Chen, Yinbo and Liu, Sifei and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2012.09161},
  year={2020}
}
```

## Results

|                                                      method                                                      | scale | Set5<br>PSNR / SSIM | Set14<br>PSNR / SSIM | DIV2K <br>PSNR / SSIM |                                                                                                                           Download                                                                                                                            |
| :--------------------------------------------------------------------------------------------------------------: | :---: | :-----------------: | :------------------: | :-------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [liif_edsr_norm_x2-4_c64b16_g1_1000k_div2k](/configs/restorers/liif/liif_edsr_norm_x2-4_c64b16_g1_1000k_div2k.py) |  x2   |  35.7148 / 0.9367   |   31.5936 / 0.8889   |   34.5896 / 0.9352    | [model](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210319-329ce255.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210319-329ce255.log.json) |
|                                                        △                                                         |  x3   |  32.3596 / 0.8914   |   28.4475 / 0.8040   |   30.9154 / 0.8720    |                                                                                                                               △                                                                                                                               |
|                                                        △                                                         |  x4   |  30.2583 / 0.8513   |   26.7867 / 0.7377   |   29.0048 / 0.8183    |                                                                                                                               △                                                                                                                               |
|     [liif_edsr_norm_c64b16_g1_1000k_div2k](/configs/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k.py)      |  x2   |  35.7120 / 0.9365   |   31.6106 / 0.8891   |   34.6401 / 0.9353    | [model](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210319-329ce255.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210319-329ce255.log.json) |
|                                                        △                                                         |  x3   |  32.3655 / 0.8913   |   28.4605 / 0.8039   |   30.9597 / 0.8711    |                                                                                                                               △                                                                                                                               |
|                                                        △                                                         |  x4   |  30.2668 / 0.8511   |   26.8093 / 0.7377   |   29.0059 / 0.8183    |                                                                                                                               △                                                                                                                               |
|                                                        △                                                         |  x6   |  27.0907 / 0.7775   |   24.7129 / 0.6438   |   26.7694 / 0.7422    |                                                                                                                               △                                                                                                                               |
|                                                        △                                                         |  x12  |  22.9046 / 0.6255   |   21.5378 / 0.5088   |   23.7269 / 0.6373    |                                                                                                                               △                                                                                                                               |
|                                                        △                                                         |  x18  |  20.8445 / 0.5390   |   20.0215 / 0.4521   |   22.1920 / 0.5947    |                                                                                                                               △                                                                                                                               |
|                                                        △                                                         |  x24  |  19.7305 / 0.5033   |   19.0703 / 0.4218   |   21.2025 / 0.5714    |                                                                                                                               △                                                                                                                               |
|                                                        △                                                         |  x30  |  18.6646 / 0.4818   |   18.0210 / 0.3905   |   20.5022 / 0.5568    |                                                                                                                               △                                                                                                                               |

Note:

-   △ refers to ditto.
-   The two configs only differs in _testing pipeline_. So they use the same checkpoint.
-   Data is normalized according to [EDSR](/configs/restorers/edsr).
-   Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
