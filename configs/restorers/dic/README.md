# Learning Texture Transformer Network for Image Super-Resolution

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{ma2020deep,
  title={Deep face super-resolution with iterative collaboration between attentive recovery and landmark estimation},
  author={Ma, Cheng and Jiang, Zhenyu and Rao, Yongming and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5569--5578},
  year={2020}
}
```

## Results

Evaluated on RGB channels, `scale` pixels in each border are cropped before evaluation.

The metrics are `PSNR / SSIM`.

|                                         Method                                         | scale |       CUFED      |                                                                                                                  Download                                                                                                                   |
| :------------------------------------------------------------------------------------: | :---: | :--------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [dic_x8c48b6_g4_150k_CelebAHQ](/configs/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ.py) |   x8  | 25.2319 / 0.7422 | [model](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/dic/dic_x8c48b6_g4_150k_CelebAHQ_20210611-5d3439ca.log.json) |
