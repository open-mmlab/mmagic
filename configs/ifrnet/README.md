# IFRNet (CVPR'2022)

> [IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation](https://arxiv.org/abs/2205.14620)

> **Task**: Video Interpolation

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Existing flow-based frame interpolation methods almost all first estimate or model intermediate optical flow, and then use flow warped context features to synthesize target frame. However, they ignore the mutual promotion of intermediate optical flow and intermediate context feature. Also, their cascaded architecture can substantially increase the inference delay and model parameters, blocking them from lots of mobile and real-time applications. For the first time, we merge above separated flow estimation and context feature refinement into a single encoder-decoder based IFRNet for compactness and fast inference, where these two crucial elements can benefit from each other. Moreover, task-oriented flow distillation loss and feature space geometry consistency loss are newly proposed to promote intermediate motion estimation and intermediate feature reconstruction of IFRNet, respectively. Benchmark results demonstrate that our IFRNet not only achieves state-of-the-art VFI accuracy, but also enjoys fast inference speed and lightweight model size.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/43229734/209931150-06e1c83b-d5a4-4104-bf7b-3b22ae109d09.png" width="400"/>
</div >

## Results and models

|                           Method                            | FPS Upsampling Ratio |      PSNR      |      SSIM      | GPU Info |                                       Download                                        |
| :---------------------------------------------------------: | :------------------: | :------------: | :------------: | :------: | :-----------------------------------------------------------------------------------: |
| [ifrnet_in2out1_8xb4_vimeo](./ifrnet_in2out1_8xb4_vimeo.py) |          x2          | 35.7999(35.80) | 0.9680(0.9794) | 1 (A100) | [model](https://download.openmmlab.com/mmediting/ifrnet/IFRNet_vimeo90k-7a66b214.pth) \| log(coming soon) |
| [ifrnet_in2out7_8xb4_gopro](./ifrnet_in2out7_8xb4_gopro.py) |          x8          | 29.9394(29.84) | 0.8922(0.920)  | 1 (A100) | [model](https://download.openmmlab.com/mmediting/ifrnet/IFRNet_gopro-5d2f805a.pth) \| log(coming soon) |
| [ifrnet_in2out7_8xb4_adobe](./ifrnet_in2out7_8xb4_adobe.py) |          x8          | 30.0273(31.93) | 0.9057(0.943)  | 1 (A100) |                       model same with above \| log(coming soon)                       |
|                            Note:                            |                      |                |                |          |                                                                                       |

- a(b) where a denotes the value run by MMEditing, b denotes the value copied from the original paper.
- PSNR is evaluated on RGB channels.
- SSIM is evaluated by averaging SSIMs on RGB channels, however the original paper uses the 3D SSIM kernel.
- The evaluated images are cropped at the center of the original images with size 512 x 512.
- For adobe240fps dataset, due to lacking of the list of videos used in evaluation in original paper, we used the following videos: *720p_240fps_1*, *GOPR9635*, *GOPR9637a*, *IMG_0004a*, *IMG_0015*, *IMG_0023*, *IMG_0179*, *IMG_0183*.

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

The model for training will be released soon.

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/ifrnet/ifrnet_in2out7_8xb4_gopro.py /path/to/checkpoint

# single-gpu test
python tools/test.py configs/ifrnet/ifrnet_in2out7_8xb4_gopro.py /path/to/checkpoint

# multi-gpu test
./tools/dist_test.sh configs/ifrnet/ifrnet_in2out7_8xb4_gopro.py /path/to/checkpoint 8
```

Pretrained checkpoints will come soon.

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](../../docs/en/user_guides/train_test.md).

</details>

## Citation

```bibtex
@InProceedings{Kong_2022_CVPR,
  author = {Kong, Lingtong and Jiang, Boyuan and Luo, Donghao and Chu, Wenqing and Huang, Xiaoming and Tai, Ying and Wang, Chengjie and Yang, Jie},
  title = {IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```
