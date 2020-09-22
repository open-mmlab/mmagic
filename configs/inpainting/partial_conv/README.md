# Image inpainting for Irregular Holes using Partial Convolutions

## Introduction

```
@inproceedings{liu2018image,
  title={Image inpainting for irregular holes using partial convolutions},
  author={Liu, Guilin and Reda, Fitsum A and Shih, Kevin J and Wang, Ting-Chun and Tao, Andrew and Catanzaro, Bryan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={85--100},
  year={2018}
}
```

## Results and models
### Places365-Challenge
| Method | Mask Type | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  |                                                                                                                        Download                                                                                                                         |
| :----: | :-------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PConv  | free-form |  256x256   |    500k     | Places365-val |  8.776   | 22.762 | 0.801 | [model](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_places_20200619-1ffed0e8.log.json) |


### CelebA-HQ
| Method | Mask Type | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  |                                                                                                                        Download                                                                                                                         |
| :----: | :-------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PConv  | free-form |  256x256   |    500k     | CelebA-val |  5.990   | 25.404 | 0.853 | [model](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/pconv/pconv_256x256_stage2_4x2_celeba_20200619-860f8b95.log.json) |
