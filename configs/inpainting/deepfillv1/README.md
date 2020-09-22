# Generative Image Inpainting with Contextual Attention

## Introduction

```
@inproceedings{yu2018generative,
  title={Generative image inpainting with contextual attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5505--5514},
  year={2018}
}
```


## Results and models
### Places365-Challenge
|   Method   |  Mask Type  | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  |                                                                                                                           Download                                                                                                                            |
| :--------: | :---------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DeepFillv1 | square bbox |  256x256   |    3500k    | Places365-val |  11.019  | 23.429 | 0.862 | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_8x2_places_20200619-c00a0e21.log.json) |


### CelebA-HQ
|   Method   |  Mask Type  | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  |                                                                                                                           Download                                                                                                                            |
| :--------: | :---------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DeepFillv1 | square bbox |  256x256   |    1500k    | CelebA-val |  6.677   | 26.878 | 0.911 | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv1/deepfillv1_256x256_4x4_celeba_20200619-dd51a855.log.json) |
