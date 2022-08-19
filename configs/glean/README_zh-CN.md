# GLEAN (CVPR'2021)

<!-- [ALGORITHM] -->

<details>
<summary align="right">GLEAN (CVPR'2021)</summary>

```bibtex
@InProceedings{chan2021glean,
  author = {Chan, Kelvin CK and Wang, Xintao and Xu, Xiangyu and Gu, Jinwei and Loy, Chen Change},
  title = {GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

</details>

<br/>

有关训练和测试中使用的元信息，请参阅[此处](https://github.com/ckkelvinchan/GLEAN)。 结果在 RGB 通道上进行评估。

|                                                   算法                                                    | PSNR  |         GPU 信息         |                                                                                                                                  下载                                                                                                                                   |
| :-------------------------------------------------------------------------------------------------------: | :---: | :----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                            [glean_cat_8x](/configs/glean/glean_2xb8_cat-x8.py)                            | 23.98 | 2 (Tesla V100-PCIE-32GB) |                              [模型](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614-d3ac8683.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_8x_20210614_145540.log.json)                              |
|                          [glean_ffhq_16x](/configs/glean/glean_2xb8_ffhq-x16.py)                          | 26.91 | 2 (Tesla V100-PCIE-32GB) |                            [模型](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527-61a3afad.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/glean/glean_ffhq_16x_20210527_194536.log.json)                            |
|                           [glean_cat_16x](/configs/glean/glean_2xb8_cat-x16.py)                           | 20.88 | 2 (Tesla V100-PCIE-32GB) |                             [模型](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527-68912543.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/glean/glean_cat_16x_20210527_103708.log.json)                             |
| [glean_in128out1024_4x2_300k_ffhq_celebahq](/configs/glean/glean_in128out1024_300k-4xb2_ffhq-celebahq.py) | 27.94 | 4 (Tesla V100-SXM3-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812_100549.log.json) |
