# DIC (CVPR'2020)

<!-- [ALGORITHM] -->

<details>

<summary align="right">DIC (CVPR'2020)</summary>

```bibtex
@inproceedings{ma2020deep,
  title={Deep face super-resolution with iterative collaboration between attentive recovery and landmark estimation},
  author={Ma, Cheng and Jiang, Zhenyu and Rao, Yongming and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5569--5578},
  year={2020}
}
```

</details>

<br/>

在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
我们使用 `PSNR` 和 `SSIM` 作为指标。

在 `dic_gan_x8c48b6_g4_150k_CelebAHQ` 的日志中，DICGAN 在 CelebA-HQ 测试集的前9张图片上进行了验证，因此下表中的 `PSNR/SSIM` 与日志数据不同。

|                                           算法                                            | scale |    CelebA-HQ     |                                            下载                                            |
| :---------------------------------------------------------------------------------------: | :---: | :--------------: | :----------------------------------------------------------------------------------------: |
|  [dic_x8c48b6_g4_150k_CelebAHQ](/configs/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ.py)   |  x8   | 25.2319 / 0.7422 | [模型](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.log.json) |
| [dic_gan_x8c48b6_g4_150k_CelebAHQ](/configs/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ.py) |  x8   | 23.6241 / 0.6721 | [模型](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.log.json) |
