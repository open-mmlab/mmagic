# PConv (ECCV'2018)

<!-- [ALGORITHM] -->

<details>
<summary align="right">PConv (ECCV'2018)</summary>

```bibtex
@inproceedings{liu2018image,
  title={Image inpainting for irregular holes using partial convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={85--100},
  year={2018}
}
```

</details>

<br/>

**Places365-Challenge**

|                                  算法                                   | 掩膜类型  | 分辨率  | 训练集容量 |    测试集     | l1 损失 |  PSNR  | SSIM  |                                   下载                                   |
| :---------------------------------------------------------------------: | :-------: | :-----: | :--------: | :-----------: | :-----: | :----: | :---: | :----------------------------------------------------------------------: |
| [PConv](/configs/inpainting/partial_conv/pconv_256x256_stage2_4x2_places.py) | free-form | 256x256 |    500k    | Places365-val |  8.776  | 22.762 | 0.801 | [模型](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.log.json) |

**CelebA-HQ**

|                                   算法                                    | 掩膜类型  | 分辨率  | 训练集容量 |   测试集   | l1 损失 |  PSNR  | SSIM  |                                   下载                                    |
| :-----------------------------------------------------------------------: | :-------: | :-----: | :--------: | :--------: | :-----: | :----: | :---: | :-----------------------------------------------------------------------: |
| [PConv](/configs/inpainting/partial_conv/pconv_256x256_stage2_4x2_celeba.py) | free-form | 256x256 |    500k    | CelebA-val |  5.990  | 25.404 | 0.853 | [模型](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.log.json) |
