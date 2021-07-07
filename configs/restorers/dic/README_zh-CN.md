# DIC (CVPR'2020)

<!-- [ALGORITHM] -->
<details>

<summary align="right">DIC (CVPR'2020)</summary>

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

|                                             Method                                             | scale |     CelebA-HQ    |                                                                                                                      Download                                                                                                                       |
| :--------------------------------------------------------------------------------------------: | :---: | :--------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [dic_x8c48b6_g4_150k_CelebAHQ](/configs/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ.py)         |   x8  | 25.2319 / 0.7422 | [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.log.json)         |
| [dic_gan_x8c48b6_g4_150k_CelebAHQ](/configs/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ.py) |   x8  | 23.6241 / 0.6721 | [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_gan_x8c48b6_g4_500k_CelebAHQ_20210625-3b89a358.log.json) |
