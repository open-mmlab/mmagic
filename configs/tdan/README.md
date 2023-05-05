# TDAN (CVPR'2020)

> [TDAN: Temporally Deformable Alignment Network for Video Super-Resolution](https://arxiv.org/abs/1812.02898)

> **Task**: Video Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video super-resolution (VSR) aims to restore a photo-realistic high-resolution (HR) video frame from both its corresponding low-resolution (LR) frame (reference frame) and multiple neighboring frames (supporting frames). Due to varying motion of cameras or objects, the reference frame and each support frame are not aligned. Therefore, temporal alignment is a challenging yet important problem for VSR. Previous VSR methods usually utilize optical flow between the reference frame and each supporting frame to wrap the supporting frame for temporal alignment. Therefore, the performance of these image-level wrapping-based models will highly depend on the prediction accuracy of optical flow, and inaccurate optical flow will lead to artifacts in the wrapped supporting frames, which also will be propagated into the reconstructed HR video frame. To overcome the limitation, in this paper, we propose a temporal deformable alignment network (TDAN) to adaptively align the reference frame and each supporting frame at the feature level without computing optical flow. The TDAN uses features from both the reference frame and each supporting frame to dynamically predict offsets of sampling convolution kernels. By using the corresponding kernels, TDAN transforms supporting frames to align with the reference frame. To predict the HR video frame, a reconstruction network taking aligned frames and the reference frame is utilized. Experimental results demonstrate the effectiveness of the proposed TDAN-based VSR model.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/144035224-a87cc41e-1352-4ffa-8b07-eda5ace8a0b1.png" width="400"/>
</div >

## Results and models

Evaluated on Y-channel. 8 pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                 Model                                  |     Dataset     | PSNR (Y)  | SSIM (Y)  |    Training Resources    |                                 Download                                  |
| :--------------------------------------------------------------------: | :-------------: | :-------: | :-------: | :----------------------: | :-----------------------------------------------------------------------: |
| [tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi](./tdan_x4_8xb16-lr1e-4-400k_vimeo90k-bi.py) |        -        |     -     |     -     | 8 (Tesla V100-SXM2-32GB) |                                     -                                     |
| [tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bd](./tdan_x4_8xb16-lr1e-4-400k_vimeo90k-bd.py) |        -        |     -     |     -     | 8 (Tesla V100-SXM2-32GB) |                                     -                                     |
| [tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi](./tdan_x4ft_8xb16-lr5e-5-400k_vimeo90k-bi.py) |   Vid4 (BIx4)   | **26.49** | **0.792** | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi](./tdan_x4ft_8xb16-lr5e-5-400k_vimeo90k-bi.py) | SPMCS-30 (BIx4) | **30.42** | **0.856** | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi](./tdan_x4ft_8xb16-lr5e-5-400k_vimeo90k-bi.py) |   Vid4 (BDx4)   |   25.93   |   0.772   | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi](./tdan_x4ft_8xb16-lr5e-5-400k_vimeo90k-bi.py) | SPMCS-30 (BDx4) |   29.69   |   0.842   | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528_135616.log.json) |
| [tdan_x4ft_1xb16-lr5e-5-800k_vimeo90k-bd](./tdan_x4ft_8xb16-lr5e-5-800k_vimeo90k-bd.py) |   Vid4 (BIx4)   |   25.80   |   0.784   | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |
| [tdan_x4ft_1xb16-lr5e-5-800k_vimeo90k-bd](./tdan_x4ft_8xb16-lr5e-5-800k_vimeo90k-bd.py) | SPMCS-30 (BIx4) |   29.56   |   0.851   | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |
| [tdan_x4ft_1xb16-lr5e-5-800k_vimeo90k-bd](./tdan_x4ft_8xb16-lr5e-5-800k_vimeo90k-bd.py) |   Vid4 (BDx4)   | **26.87** | **0.815** | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |
| [tdan_x4ft_1xb16-lr5e-5-800k_vimeo90k-bd](./tdan_x4ft_8xb16-lr5e-5-800k_vimeo90k-bd.py) | SPMCS-30 (BDx4) | **30.77** | **0.868** | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528-c53ab844.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bdx4_20210528_122401.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

TDAN is trained with two stages.

**Stage 1**: Train with a larger learning rate (1e-4)

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/tdan/tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi.py

# single-gpu train
python tools/train.py configs/tdan/tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi.py

# multi-gpu train
./tools/dist_train.sh cconfigs/tdan/tdan_x4_1xb16-lr1e-4-400k_vimeo90k-bi.py 8
```

**Stage 2**: Fine-tune with a smaller learning rate (5e-5)

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py

# single-gpu train
python tools/train.py configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py

# multi-gpu train
./tools/dist_train.sh configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth

# single-gpu test
python tools/test.py configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth

# multi-gpu test
./tools/dist_test.sh configs/tdan/tdan_x4ft_1xb16-lr5e-5-400k_vimeo90k-bi.py https://download.openmmlab.com/mmediting/restorers/tdan/tdan_vimeo90k_bix4_20210528-739979d9.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@InProceedings{tian2020tdan,
  title={TDAN: Temporally-Deformable Alignment Network for Video Super-Resolution},
  author={Tian, Yapeng and Zhang, Yulun and Fu, Yun and Xu, Chenliang},
  booktitle = {Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  year = {2020}
}
```
