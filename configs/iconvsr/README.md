# IconVSR (CVPR'2021)

> [BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond](https://arxiv.org/abs/2012.02181)

> **Task**: Video Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video super-resolution (VSR) approaches tend to have more components than the image counterparts as they need to exploit the additional temporal dimension. Complex designs are not uncommon. In this study, we wish to untangle the knots and reconsider some most essential components for VSR guided by four basic functionalities, i.e., Propagation, Alignment, Aggregation, and Upsampling. By reusing some existing components added with minimal redesigns, we show a succinct pipeline, BasicVSR, that achieves appealing improvements in terms of speed and restoration quality in comparison to many state-of-the-art algorithms. We conduct systematic analysis to explain how such gain can be obtained and discuss the pitfalls. We further show the extensibility of BasicVSR by presenting an information-refill mechanism and a coupled propagation scheme to facilitate information aggregation. The BasicVSR and its extension, IconVSR, can serve as strong baselines for future VSR approaches.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144011348-c58101d4-5f69-4e58-be2b-7accd07b06fa.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels for REDS4 and Y channel for others. The metrics are `PSNR` / `SSIM` .
The pretrained weights of the IconVSR components can be found here: [SPyNet](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth), [EDVR-M for REDS](https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_reds_20210413-3867262f.pth), and [EDVR-M for Vimeo-90K](https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_vimeo90k_20210413-e40e99a8.pth).

|       Method        | REDS4 (BIx4) PSNR (RGB) | Vimeo-90K-T (BIx4) PSNR (Y) | Vid4 (BIx4) PSNR (Y) | UDM10 (BDx4) PSNR (Y) | Vimeo-90K-T (BDx4) PSNR (Y) | Vid4 (BDx4) PSNR (Y) |       GPU Info        |        Download        |
| :-----------------: | :---------------------: | :-------------------------: | :------------------: | :-------------------: | :-------------------------: | :------------------: | :-------------------: | :--------------------: |
| [iconvsr_reds4](./iconvsr_2xb4_reds4.py) |       **31.6926**       |           36.4983           |     **27.4809**      |        35.3377        |           34.4299           |       25.2110        | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413_222735.log.json) |
| [iconvsr_vimeo90k_bi](./iconvsr_2xb4_vimeo90k-bi.py) |         30.3452         |         **37.3729**         |       27.4238        |        34.2595        |           34.5548           |       24.6666        | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413-7c7418dc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413_222757.log.json) |
| [iconvsr_vimeo90k_bd](./iconvsr_2xb4_vimeo90k-bd.py) |         29.0150         |           34.6780           |       26.3109        |      **40.0640**      |         **37.7573**         |     **28.2464**      | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414-5f38cb34.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414_084128.log.json) |

|       Method        | REDS4 (BIx4) SSIM (RGB) | Vimeo-90K-T (BIx4) SSIM (Y) | Vid4 (BIx4) SSIM (Y) | UDM10 (BDx4) SSIM (Y) | Vimeo-90K-T (BDx4) SSIM (Y) | Vid4 (BDx4) SSIM (Y) |       GPU Info        |        Download        |
| :-----------------: | :---------------------: | :-------------------------: | :------------------: | :-------------------: | :-------------------------: | :------------------: | :-------------------: | :--------------------: |
| [iconvsr_reds4](./iconvsr_2xb4_reds4.py) |       **0.8951**        |           0.9416            |      **0.8354**      |        0.9471         |           0.9287            |        0.7732        | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413_222735.log.json) |
| [iconvsr_vimeo90k_bi](./iconvsr_2xb4_vimeo90k-bi.py) |         0.8659          |         **0.9467**          |        0.8297        |        0.9398         |           0.9295            |        0.7491        | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413-7c7418dc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413_222757.log.json) |
| [iconvsr_vimeo90k_bd](./iconvsr_2xb4_vimeo90k-bd.py) |         0.8465          |           0.9339            |        0.8028        |      **0.9697**       |         **0.9517**          |      **0.8612**      | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414-5f38cb34.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414_084128.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/iconvsr/iconvsr_2xb4_reds4.py

# single-gpu train
python tools/train.py configs/iconvsr/iconvsr_2xb4_reds4.py

# multi-gpu train
./tools/dist_train.sh configs/iconvsr/iconvsr_2xb4_reds4.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/iconvsr/iconvsr_2xb4_reds4.py https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth

# single-gpu test
python tools/test.py configs/iconvsr/iconvsr_2xb4_reds4.py https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth

# multi-gpu test
./tools/dist_test.sh configs/iconvsr/iconvsr_2xb4_reds4.py https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>

## Citation

```bibtex
@InProceedings{chan2021basicvsr,
  author = {Chan, Kelvin CK and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title = {BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```
