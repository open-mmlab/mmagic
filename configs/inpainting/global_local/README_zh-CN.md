# Global&Local (ToG'2017)

<!-- [ALGORITHM] -->

<details>
<summary align="right">Global&Local (ToG'2017)</summary>

```bibtex
@article{iizuka2017globally,
  title={Globally and locally consistent image completion},
  author={Iizuka, Satoshi and Simo-Serra, Edgar and Ishikawa, Hiroshi},
  journal={ACM Transactions on Graphics (ToG)},
  volume={36},
  number={4},
  pages={1--14},
  year={2017},
  publisher={ACM New York, NY, USA}
}
```

</details>

<br/>

*请注意，为了与当前的深度图像修复方法进行公平比较，我们没有在 Global&Local 中使用后处理模块。*

**Places365-Challenge**

|                                  算法                                  |  掩膜类型   | 分辨率  | 训练集容量 |    测试集     | l1 损失 |  PSNR  | SSIM  |                                  下载                                   |
| :--------------------------------------------------------------------: | :---------: | :-----: | :--------: | :-----------: | :-----: | :----: | :---: | :---------------------------------------------------------------------: |
| [Global&Local](/configs/inpainting/global_local/gl_256x256_8x12_places.py) | square bbox | 256x256 |    500k    | Places365-val | 11.164  | 23.152 | 0.862 | [模型](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.log.json) |

**CelebA-HQ**

|                                   算法                                   |  掩膜类型   | 分辨率  | 训练集容量 |   测试集   | l1 损失 |  PSNR  | SSIM  |                                   下载                                   |
| :----------------------------------------------------------------------: | :---------: | :-----: | :--------: | :--------: | :-----: | :----: | :---: | :----------------------------------------------------------------------: |
| [Global&Local](/configs/inpainting/global_local/gl_256x256_8x12_celeba.py) | square bbox | 256x256 |    500k    | CelebA-val |  6.678  | 26.780 | 0.904 | [模型](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth) \| [日志](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.log.json) |
