# DeepFillv1 (CVPR'2018)

<!-- [ALGORITHM] -->

<details>
<summary align="right">DeepFillv1 (CVPR'2018)</summary>

```bibtex
@inproceedings{yu2018generative,
  title={Generative image inpainting with contextual attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5505--5514},
  year={2018}
}
```

</details>

<br/>

**Places365-Challenge**

|                                  算法                                  |  掩膜类型   | 分辨率  | 训练集容量 |    测试集     | l1 损失 |  PSNR  | SSIM  |                                  下载                                   |
| :--------------------------------------------------------------------: | :---------: | :-----: | :--------: | :-----------: | :-----: | :----: | :---: | :---------------------------------------------------------------------: |
| [DeepFillv1](/configs/inpainting/deepfillv1/deepfillv1_256x256_8x2_places.py) | square bbox | 256x256 |   3500k    | Places365-val | 11.019  | 23.429 | 0.862 | [模型](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.log.json) |

**CelebA-HQ**

|                                   算法                                   |  掩膜类型   | 分辨率  | 训练集容量 |   测试集   | l1 损失 |  PSNR  | SSIM  |                                   下载                                   |
| :----------------------------------------------------------------------: | :---------: | :-----: | :--------: | :--------: | :-----: | :----: | :---: | :----------------------------------------------------------------------: |
| [DeepFillv1](/configs/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba.py) | square bbox | 256x256 |   1500k    | CelebA-val |  6.677  | 26.878 | 0.911 | [模型](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.log.json) |
