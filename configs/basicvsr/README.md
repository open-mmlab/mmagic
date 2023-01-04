# BasicVSR (CVPR'2021)

> [BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond](https://arxiv.org/abs/2012.02181)

> **Task**: Video Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video super-resolution (VSR) approaches tend to have more components than the image counterparts as they need to exploit the additional temporal dimension. Complex designs are not uncommon. In this study, we wish to untangle the knots and reconsider some most essential components for VSR guided by four basic functionalities, i.e., Propagation, Alignment, Aggregation, and Upsampling. By reusing some existing components added with minimal redesigns, we show a succinct pipeline, BasicVSR, that achieves appealing improvements in terms of speed and restoration quality in comparison to many state-of-the-art algorithms. We conduct systematic analysis to explain how such gain can be obtained and discuss the pitfalls. We further show the extensibility of BasicVSR by presenting an information-refill mechanism and a coupled propagation scheme to facilitate information aggregation. The BasicVSR and its extension, IconVSR, can serve as strong baselines for future VSR approaches.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144011085-fdded077-24de-468b-826e-5f82716219a5.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels for REDS4 and Y channel for others. The metrics are `PSNR` / `SSIM` .
The pretrained weights of SPyNet can be found [here](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth).

|                         Method                         | REDS4 (BIx4) PSNR (RGB) | Vimeo-90K-T (BIx4) PSNR (Y) | Vid4 (BIx4) PSNR (Y) | UDM10 (BDx4) PSNR (Y) | Vimeo-90K-T (BDx4) PSNR (Y) | Vid4 (BDx4) PSNR (Y) | REDS4 (BIx4) SSIM (RGB) | Vimeo-90K-T (BIx4) SSIM (Y) | Vid4 (BIx4) SSIM (Y) | UDM10 (BDx4) SSIM (Y) | Vimeo-90K-T (BDx4) SSIM (Y) | Vid4 (BDx4) SSIM (Y) |         GPU Info         |                                                                                                              Download                                                                                                               |
| :----------------------------------------------------: | :---------------------: | :-------------------------: | :------------------: | :-------------------: | :-------------------------: | :------------------: | :---------------------: | :-------------------------: | :------------------: | :-------------------: | :-------------------------: | :------------------: | :----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [basicvsr_reds4](./basicvsr_2xb4_reds4.py)       |       **31.4170**       |           36.2848           |       27.2694        |        33.4478        |           34.4700           |       24.4541        |       **0.8909**        |           0.9395            |        0.8318        |        0.9306         |           0.9286            |        0.7455        | 2 (Tesla V100-PCIE-32GB) |       [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20210409_092646.log.json)       |
| [basicvsr_vimeo90k_bi](./basicvsr_2xb4_vimeo90k-bi.py) |         30.3128         |         **37.2026**         |     **27.2755**      |        34.5554        |           34.8097           |       25.0517        |         0.8660          |         **0.9451**          |      **0.8248**      |        0.9434         |           0.9316            |        0.7636        | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409-d2d8f760.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409_132702.log.json) |
| [basicvsr_vimeo90k_bd](./basicvsr_2xb4_vimeo90k-bd.py) |         29.0376         |           34.6427           |       26.2708        |      **39.9953**      |         **37.5501**         |     **27.9791**      |         0.8481          |           0.9335            |        0.8022        |      **0.9695**       |         **0.9499**          |      **0.8556**      | 2 (Tesla V100-PCIE-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409-0154dd64.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409_132740.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/basicvsr/basicvsr_2xb4_reds4.py

# single-gpu train
python tools/train.py configs/basicvsr/basicvsr_2xb4_reds4.py

# multi-gpu train
./tools/dist_train.sh configs/basicvsr/basicvsr_2xb4_reds4.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/basicvsr/basicvsr_2xb4_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth

# single-gpu test
python tools/test.py configs/basicvsr/basicvsr_2xb4_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth

# multi-gpu test
./tools/dist_test.sh configs/basicvsr/basicvsr_2xb4_reds4.py https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth 8
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
