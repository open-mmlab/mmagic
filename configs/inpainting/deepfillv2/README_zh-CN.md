# DeepFillv2 (CVPR'2019)

<!-- [ALGORITHM] -->

<details>
<summary align="right">DeepFillv2 (CVPR'2019)</summary>

```bibtex
@inproceedings{yu2019free,
  title={Free-form image inpainting with gated convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4471--4480},
  year={2019}
}
```

</details>

<br/>

**Places365-Challenge**

|                                  算法                                   | 掩膜类型  | 分辨率  | 训练集容量 |    测试集     | l1 损失 |  PSNR  | SSIM  |                                   下载                                   |
| :---------------------------------------------------------------------: | :-------: | :-----: | :--------: | :-----------: | :-----: | :----: | :---: | :----------------------------------------------------------------------: |
| [DeepFillv2](/configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_places.py) | free-form | 256x256 |    100k    | Places365-val |  8.635  | 22.398 | 0.815 | [模型](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.log.json) |

**CelebA-HQ**

|                                   算法                                    | 掩膜类型  | 分辨率  | 训练集容量 |   测试集   | l1 损失 |  PSNR  | SSIM  |                                   下载                                    |
| :-----------------------------------------------------------------------: | :-------: | :-----: | :--------: | :--------: | :-----: | :----: | :---: | :-----------------------------------------------------------------------: |
| [DeepFillv2](/configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba.py) | free-form | 256x256 |    20k     | CelebA-val |  5.411  | 25.721 | 0.871 | [模型](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.log.json) |
