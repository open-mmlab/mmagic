# DIC (CVPR'2020)

<!-- [ALGORITHM] -->
<details>

<summary align="right"><a href="https://arxiv.org/abs/2003.13063">DIC (CVPR'2020)</a></summary>

```bibtex
@inproceedings{ma2020deep,
  title={Deep face super-resolution with iterative collaboration between attentive recovery and landmark estimation},
  author={Ma, Cheng and Jiang, Zhenyu and Rao, Yongming and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5569--5578},
  year={2020}
}
```

</details>

<br/>

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.
The metrics are `PSNR / SSIM` .

In the log data of `dic_gan_x8c48b6_g4_150k_CelebAHQ`, DICGAN is verified on the first 9 pictures of the test set of CelebA-HQ, so `PSNR/SSIM` shown in the follow table is different from the log data.

|                                             Method                                         | scale |     CelebA-HQ    |                                                                                                                      Download                                                                                                                   |
| :----------------------------------------------------------------------------------------: | :---: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [dic_x8c48b6_g4_150k_celeba](/configs/restorers/dic/dic_x8c48b6_g4_150k_celeba.py)         |   x8  | 25.0673 / 0.7341 | [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_celeba_20211010-2fc30d24.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_celeba_20211010-2fc30d24.log.json)         |
| [dic_gan_x8c48b6_g4_150k_celeba](/configs/restorers/dic/dic_gan_x8c48b6_g4_500k_celeba.py) |   x8  | 24.0153 / 0.6883 | [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_celeba_20211010-f5178387.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_celeba_20211010-f5178387.log.json) |
