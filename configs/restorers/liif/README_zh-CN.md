# LIIF (CVPR'2021)

<!-- [ALGORITHM] -->

<details>
<summary align="right">LIIF (CVPR'2021)</summary>

```bibtex
@inproceedings{chen2021learning,
  title={Learning continuous image representation with local implicit image function},
  author={Chen, Yinbo and Liu, Sifei and Wang, Xiaolong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8628--8638},
  year={2021}
}
```

</details>

<br/>

|                                算法                                 | scale | Set5<br>PSNR / SSIM | Set14<br>PSNR / SSIM | DIV2K <br>PSNR / SSIM |                                 下载                                 |
| :-----------------------------------------------------------------: | :---: | :-----------------: | :------------------: | :-------------------: | :------------------------------------------------------------------: |
| [liif_edsr_norm_x2-4_c64b16_g1_1000k_div2k](/configs/restorers/liif/liif_edsr_norm_x2-4_c64b16_g1_1000k_div2k.py) |  x2   |  35.7148 / 0.9367   |   31.5936 / 0.8889   |   34.5896 / 0.9352    | [模型](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210319-329ce255.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210319-329ce255.log.json) |
|                                  △                                  |  x3   |  32.3596 / 0.8914   |   28.4475 / 0.8040   |   30.9154 / 0.8720    |                                  △                                   |
|                                  △                                  |  x4   |  30.2583 / 0.8513   |   26.7867 / 0.7377   |   29.0048 / 0.8183    |                                  △                                   |
| [liif_edsr_norm_c64b16_g1_1000k_div2k](/configs/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k.py) |  x2   |  35.7120 / 0.9365   |   31.6106 / 0.8891   |   34.6401 / 0.9353    | [模型](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210319-329ce255.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210319-329ce255.log.json) |
|                                  △                                  |  x3   |  32.3655 / 0.8913   |   28.4605 / 0.8039   |   30.9597 / 0.8711    |                                  △                                   |
|                                  △                                  |  x4   |  30.2668 / 0.8511   |   26.8093 / 0.7377   |   29.0059 / 0.8183    |                                  △                                   |
|                                  △                                  |  x6   |  27.0907 / 0.7775   |   24.7129 / 0.6438   |   26.7694 / 0.7422    |                                  △                                   |
|                                  △                                  |  x12  |  22.9046 / 0.6255   |   21.5378 / 0.5088   |   23.7269 / 0.6373    |                                  △                                   |
|                                  △                                  |  x18  |  20.8445 / 0.5390   |   20.0215 / 0.4521   |   22.1920 / 0.5947    |                                  △                                   |
|                                  △                                  |  x24  |  19.7305 / 0.5033   |   19.0703 / 0.4218   |   21.2025 / 0.5714    |                                  △                                   |
|                                  △                                  |  x30  |  18.6646 / 0.4818   |   18.0210 / 0.3905   |   20.5022 / 0.5568    |                                  △                                   |

注：

- △ 指同上。
- 这两个配置仅在 _testing pipeline_ 上有所不同。 所以他们使用相同的检查点。
- 数据根据 [EDSR](/configs/restorers/edsr) 进行正则化。
- 在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。
