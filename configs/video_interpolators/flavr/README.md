# FLAVR (arXiv'2020)

> [FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation](https://arxiv.org/pdf/2012.08512.pdf)

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

|                                          Method                                          | scale | Vimeo90k-triplet  |                                          Download                                          |
| :--------------------------------------------------------------------------------------: | :---: | :---------------: | :----------------------------------------------------------------------------------------: |
| [flavr_in4out1_g8b4_vimeo90k_septuplet](/configs/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet.py) |  x2   | 36.3340 / 0.96015 | [model](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth) \| [log](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.log.json) |

Note: FLAVR for x8 VFI task will supported in the future.

## Citation

```bibtex
@article{kalluri2020flavr,
  title={Flavr: Flow-agnostic video representations for fast frame interpolation},
  author={Kalluri, Tarun and Pathak, Deepak and Chandraker, Manmohan and Tran, Du},
  journal={arXiv preprint arXiv:2012.08512},
  year={2020}
}
```
