# LIIF (CVPR'2021)

> **任务**: 图像超分辨率

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

|                             算法                              | scale | Set5<br>PSNR / SSIM | Set14<br>PSNR / SSIM | DIV2K <br>PSNR / SSIM |   GPU 信息   |                              下载                              |
| :-----------------------------------------------------------: | :---: | :-----------------: | :------------------: | :-------------------: | :----------: | :------------------------------------------------------------: |
| [liif_edsr_norm_c64b16_g1_1000k_div2k](./liif-edsr-norm_c64b16_1xb16-1000k_div2k.py) |  x2   |  35.7131 / 0.9366   |   31.5579 / 0.8889   |   34.6647 / 0.9355    | 1 (TITAN Xp) | [模型](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.log.json) |
|                               △                               |  x3   |  32.3805 / 0.8915   |   28.4605 / 0.8039   |   30.9808 / 0.8724    |      △       |                               △                                |
|                               △                               |  x4   |  30.2748 / 0.8509   |   26.8415 / 0.7381   |   29.0245 / 0.8187    |      △       |                               △                                |
|                               △                               |  x6   |  27.1187 / 0.7774   |   24.7461 / 0.6444   |   26.7770 / 0.7425    |      △       |                               △                                |
|                               △                               |  x18  |  20.8516 / 0.5406   |   20.0096 / 0.4525   |   22.1987 / 0.5955    |      △       |                               △                                |
|                               △                               |  x30  |  18.8467 / 0.5010   |   18.1321 / 0.3963   |   20.5050 / 0.5577    |      △       |                               △                                |
| [liif_rdn_norm_c64b16_g1_1000k_div2k](./liif-rdn-norm_c64b16_1xb16-1000k_div2k.py) |  x2   |  35.7874 / 0.9366   |   31.6866 / 0.8896   |   34.7548 / 0.9356    | 1 (TITAN Xp) | [模型](https://download.openmmlab.com/mmediting/restorers/liif/liif_rdn_norm_c64b16_g1_1000k_div2k_20210717-22d6fdc8.pth) \| [日志](https://download.openmmlab.com/mmediting/restorers/liif/liif_rdn_norm_c64b16_g1_1000k_div2k_20210717-22d6fdc8.log.json) |
|                               △                               |  x3   |  32.4992 / 0.8923   |   28.4905 / 0.8037   |   31.0744 / 0.8731    |      △       |                               △                                |
|                               △                               |  x4   |  30.3835 / 0.8513   |   26.8734 / 0.7373   |   29.1101 / 0.8197    |      △       |                               △                                |
|                               △                               |  x6   |  27.1914 / 0.7751   |   24.7824 / 0.6434   |   26.8693 / 0.7437    |      △       |                               △                                |
|                               △                               |  x18  |  20.8913 / 0.5329   |   20.1077 / 0.4537   |   22.2972 / 0.5950    |      △       |                               △                                |
|                               △                               |  x30  |  18.9354 / 0.4864   |   18.1448 / 0.3942   |   20.5663 / 0.5560    |      △       |                               △                                |

注：

- △ 指同上。
- 这两个配置仅在 _testing pipeline_ 上有所不同。 所以他们使用相同的检查点。
- 数据根据 [EDSR](../edsr/README.md) 进行正则化。
- 在 RGB 通道上进行评估，在评估之前裁剪每个边界中的 `scale` 像素。

## 快速开始

**训练**

<details>
<summary>训练说明</summary>

您可以使用以下命令来训练模型。

```shell
# CPU上训练
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py

# 单个GPU上训练
python tools/train.py configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py

# 多个GPU上训练
./tools/dist_train.sh configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Train a model** 部分。

</details>

**测试**

<details>
<summary>测试说明</summary>

您可以使用以下命令来测试模型。

```shell
# CPU上测试
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth

# 单个GPU上测试
python tools/test.py configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth

# 多个GPU上测试
./tools/dist_test.sh configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth 8
```

更多细节可以参考 [train_test.md](/docs/zh_cn/user_guides/train_test.md) 中的 **Test a pre-trained model** 部分。

</details>
