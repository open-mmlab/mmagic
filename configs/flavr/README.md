# FLAVR (arXiv'2020)

> [FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation](https://arxiv.org/pdf/2012.08512.pdf)

> **Task**: Video Interpolation

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Most modern frame interpolation approaches rely on explicit bidirectional optical flows between adjacent frames, thus are sensitive to the accuracy of underlying flow estimation in handling occlusions while additionally introducing computational bottlenecks unsuitable for efficient deployment. In this work, we propose a flow-free approach that is completely end-to-end trainable for multi-frame video interpolation. Our method, FLAVR, is designed to reason about non-linear motion trajectories and complex occlusions implicitly from unlabeled videos and greatly simplifies the process of training, testing and deploying frame interpolation models. Furthermore, FLAVR delivers up to 6× speed up compared to the current state-of-the-art methods for multi-frame interpolation while consistently demonstrating superior qualitative and quantitative results compared with prior methods on popular benchmarks including Vimeo-90K, Adobe-240FPS, and GoPro. Finally, we show that frame interpolation is a competitive self-supervised pre-training task for videos via demonstrating various novel applications of FLAVR including action recognition, optical flow estimation, motion magnification, and video object tracking. Code and trained models are provided in the supplementary material.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/56712176/169070212-52acdcea-d732-4441-9983-276e2e40b195.png" width="400"/>
</div >

## Results and models

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                   Model                                    |  Dataset   | scale |  PSNR   |  SSIM   | Training Resources  |                                    Download                                    |
| :------------------------------------------------------------------------: | :--------: | :---: | :-----: | :-----: | :-----------------: | :----------------------------------------------------------------------------: |
| [flavr_in4out1_g8b4_vimeo90k_septuplet](./flavr_in4out1_8xb4_vimeo90k-septuplet.py) | vimeo90k-T |  x2   | 36.3340 | 0.96015 | 8 (Tesla PG503-216) | [model](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.log.json) |

Note: FLAVR for x8 VFI task will supported in the future.

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py

# single-gpu train
python tools/train.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py

# multi-gpu train
./tools/dist_train.sh configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth

# single-gpu test
python tools/test.py configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth

# multi-gpu test
./tools/dist_test.sh configs/flavr/flavr_in4out1_8xb4_vimeo90k-septuplet.py https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@article{kalluri2020flavr,
  title={Flavr: Flow-agnostic video representations for fast frame interpolation},
  author={Kalluri, Tarun and Pathak, Deepak and Chandraker, Manmohan and Tran, Du},
  journal={arXiv preprint arXiv:2012.08512},
  year={2020}
}
```
