# BasicVSR (CVPR'2021)

<!-- [ALGORITHM] -->

<details>
<summary align="right">BasicVSR (CVPR'2021)</summary>

```bibtex
@InProceedings{chan2021basicvsr,
  author = {Chan, Kelvin CK and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title = {BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

</details>

<br/>

对于 REDS4，我们对 RGB 通道进行评估。对于其他数据集，我们对 Y 通道进行评估。我们使用 `PSNR` 和 `SSIM` 作为指标。
SPyNet 的 预训练权重在[这里](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth)。

|  算法   | REDS4 (BIx4)<br>PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4)<br>PSNR/SSIM (Y) | Vid4 (BIx4)<br>PSNR/SSIM (Y) | UDM10 (BDx4)<br>PSNR/SSIM (Y) | Vimeo-90K-T (BDx4)<br>PSNR/SSIM (Y) | Vid4 (BDx4)<br>PSNR/SSIM (Y) |  下载   |
| :-----: | :-----------------------------: | :---------------------------------: | :--------------------------: | :---------------------------: | :---------------------------------: | :--------------------------: | :-----: |
| [basicvsr_reds4](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds4.py) |       **31.4170/0.8909**        |           36.2848/0.9395            |        27.2694/0.8318        |        33.4478/0.9306         |           34.4700/0.9286            |        24.4541/0.7455        | [模型](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20210409_092646.log.json) |
| [basicvsr_vimeo90k_bi](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_vimeo90k_bi.py) |         30.3128/0.8660          |         **37.2026/0.9451**          |      **27.2755/0.8248**      |        34.5554/0.9434         |           34.8097/0.9316            |        25.0517/0.7636        | [模型](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409-d2d8f760.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bi_20210409_132702.log.json) |
| [basicvsr_vimeo90k_bd](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_vimeo90k_bd.py) |         29.0376/0.8481          |           34.6427/0.9335            |        26.2708/0.8022        |      **39.9953/0.9695**       |         **37.5501/0.9499**          |      **27.9791/0.8556**      | [模型](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409-0154dd64.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_vimeo90k_bd_20210409_132740.log.json) |
