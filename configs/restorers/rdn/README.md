# RDN (CVPR'2018)

<!-- [ALGORITHM] -->
<details>
<summary align="right"><a href="https://arxiv.org/abs/1802.08797">RDN (CVPR'2018)</a></summary>

```bibtex
@inproceedings{zhang2018residual,
  title={Residual dense network for image super-resolution},
  author={Zhang, Yulun and Tian, Yapeng and Kong, Yu and Zhong, Bineng and Fu, Yun},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2472--2481},
  year={2018}
}
```

</details>

## Abstract
A very deep convolutional neural network (CNN) has recently achieved great success for image super-resolution (SR) and offered hierarchical features as well. However, most deep CNN based SR models do not make full use of the hierarchical features from the original low-resolution (LR) images, thereby achieving relatively-low performance. In this paper, we propose a novel residual dense network (RDN) to address this problem in image SR. We fully exploit the hierarchical features from all the convolutional layers. Specifically, we propose residual dense block (RDB) to extract abundant local features via dense connected convolutional layers. RDB further allows direct connections from the state of preceding RDB to all the layers of current RDB, leading to a contiguous memory (CM) mechanism. Local feature fusion in RDB is then used to adaptively learn more effective features from preceding and current local features and stabilizes the training of wider network. After fully obtaining dense local features, we use global feature fusion to jointly and adaptively learn global hierarchical features in a holistic way. Extensive experiments on benchmark datasets with different degradation models show that our RDN achieves favorable performance against state-of-the-art methods.

<p align="center">
  <img src="https://user-images.githubusercontent.com/7676947/144034203-c3a4ac55-d815-4180-a345-f80ab5ca68b6.png" />
</p>

## Results and Models
Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

|                                         Method                                         |       Set5       |      Set14       |      DIV2K       |                                                                                                                   Download                                                                                                                    |
| :------------------------------------------------------------------------------------: | :--------------: | :--------------: | :--------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [rdn_x2c64b16_g1_1000k_div2k](/configs/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k.py) | 35.9883 / 0.9385 | 31.8366 / 0.8920 | 34.9392 / 0.9380 | [model](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_20210419-dc146009.pth?versionId=CAEQJxiBgMC774LGyBciIGU3ZGRkZWM3Y2Y0ZjQ2OTliZTc2NmM5ZWY0MDA1MDU3) \| [log](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x2c64b16_g1_1000k_div2k_20210419-dc146009.log.json?versionId=CAEQJxiBgICf04_HyBciIDFkMzBiY2Y2ZDE2ZDQ0ZWE4M2MxMjMwMzdhMzY1ZTUz) |
| [rdn_x3c64b16_g1_1000k_div2k](/configs/restorers/rdn/rdn_x3c64b16_g1_1000k_div2k.py) | 32.6051 / 0.8943 | 28.6338 / 0.8077 | 31.2153 / 0.8763 | [model](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x3c64b16_g1_1000k_div2k_20210419-b93cb6aa.pth?versionId=CAEQJxiBgMC3v9LFyBciIGExYWY0NTI0YWVkODQxZDRiYWNlYjViY2E5MzQ4OTc1) \| [log](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x3c64b16_g1_1000k_div2k_20210419-b93cb6aa.log.json?versionId=CAEQJxiBgMCtwtLFyBciIDNmNzZjNTUyYTk0MjQ2OTBiYjJiNDNjMTI0NGZhYmI4) |
| [rdn_x4c64b16_g1_1000k_div2k](/configs/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k.py) | 30.4922 / 0.8548 | 26.9570 / 0.7423 | 29.1925 / 0.8233 | [model](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.pth?versionId=CAEQJxiBgICwxdLFyBciIGFlMzVhNTBlOGEyNDQwMGI5OGJjOTJkMDQ1ZDJjOTM2) \| [log](https://download.openmmlab.com/mmediting/restorers/rdn/rdn_x4c64b16_g1_1000k_div2k_20210419-3577d44f.log.json?versionId=CAEQJxiBgIC9xtLFyBciIGQ5YTJhMjY0OTE1YjRiMTQ5OTc5YzQ2MjM4ZGVkZWQ1) |
