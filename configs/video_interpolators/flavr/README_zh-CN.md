# FLAVR (arXiv'2020)

> [FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation](https://arxiv.org/pdf/2012.08512.pdf)

<!-- [ALGORITHM] -->

## 预训练模型测试结果

在 RGB 通道上评估。
评估指标 `PSNR / SSIM`。

|                                           算法                                            | scale | Vimeo90k-triplet  |                                           下载                                            |
| :---------------------------------------------------------------------------------------: | :---: | :---------------: | :---------------------------------------------------------------------------------------: |
| [flavr_in4out1_g8b4_vimeo90k_septuplet](/configs/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet.py) |  x2   | 36.3340 / 0.96015 | [模型](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.pth) \| [日志](https://download.openmmlab.com/mmediting/video_interpolators/flavr/flavr_in4out1_g8b4_vimeo90k_septuplet_20220509-c2468995.log.json) |

注：FLAVR 中的 8 倍视频插帧算法将会在未来版本中支持。

## Citation

```bibtex
@article{kalluri2020flavr,
  title={Flavr: Flow-agnostic video representations for fast frame interpolation},
  author={Kalluri, Tarun and Pathak, Deepak and Chandraker, Manmohan and Tran, Du},
  journal={arXiv preprint arXiv:2012.08512},
  year={2020}
}
```
