# SRCNN (TPAMI'2015)

<!-- [ALGORITHM] -->

<details>
<summary align="right">SRCNN (TPAMI'2015)</summary>

```bibtex
@article{dong2015image,
  title={Image super-resolution using deep convolutional networks},
  author={Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={38},
  number={2},
  pages={295--307},
  year={2015},
  publisher={IEEE}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

|                                    算法                                     |       Set5       |       Set14       |      DIV2K       |                                     下载                                     |
| :-------------------------------------------------------------------------: | :--------------: | :---------------: | :--------------: | :--------------------------------------------------------------------------: |
| [srcnn_x4k915_1x16_1000k_div2k](/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py) | 28.4316 / 0.8099 | 25.6486 /  0.7014 | 27.7460 / 0.7854 | [模型](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608_120159.log.json) |
