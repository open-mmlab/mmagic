# CAIN (AAAI'2020)

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://aaai.org/ojs/index.php/AAAI/article/view/6693/6547">CAIN (AAAI'2020)</a></summary>

```bibtex
@inproceedings{choi2020channel,
  title={Channel attention is all you need for video frame interpolation},
  author={Choi, Myungsub and Kim, Heewon and Han, Bohyung and Xu, Ning and Lee, Kyoung Mu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={10663--10671},
  year={2020}
}
```

</details>

<br/>

在 RGB 通道上进行评估。
我们使用 `PSNR` 和 `SSIM` 作为指标。
学习率调整策略是等间隔调整策略。

|                                            算法                                             | vimeo-90k-triplet |                                             下载                                             |
| :-----------------------------------------------------------------------------------------: | :---------------: | :------------------------------------------------------------------------------------------: |
| [cain_b5_g1b32_vimeo90k_triplet](/configs/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet.py) | 34.6010 / 0.9578  | [模型](https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.pth)/[日志](https://download.openmmlab.com/mmediting/video_interpolators/cain/cain_b5_g1b32_vimeo90k_triplet_20220530-3520b00c.log.json) |
