# Real-ESRGAN (ICCVW'2021)

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2107.10833">Real-ESRGAN (ICCVW'2021)</a></summary>

```bibtex
@inproceedings{wang2021real,
  title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)},
  pages={1905--1914},
  year={2021}
}
```

</details>

<br/>

在 RGB 通道上进行评估，指标为 `PSNR/SSIM`。

|                                                                    算法                                                                    |      Set5      |         GPU 信息         |                                                                                                                                            下载                                                                                                                                             |
| :----------------------------------------------------------------------------------------------------------------------------------------: | :------------: | :----------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost](/configs/image_restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost.py) | 28.0297/0.8236 | 4 (Tesla V100-SXM2-32GB) |                                                                      [模型](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost_20210816-4ae3b5a4.pth)/日志                                                                      |
|  [realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost](/configs/image_restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost.py)  | 26.2204/0.7655 | 4 (Tesla V100-SXM2-32GB) | [模型](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20211010-34798885.pth) /[日志](https://download.openmmlab.com/mmediting/restorers/real_esrgan/realesrgan_c64b23g32_12x4_lr1e-4_400k_df2k_ost_20210922_142838.log.json) |
