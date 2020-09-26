# Free-form Image Inpainting with Gated Convolution

## Introduction

```
@inproceedings{yu2019free,
  title={Free-form image inpainting with gated convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4471--4480},
  year={2019}
}
```

## Results and models
### Places365-Challenge
|   Method   | Mask Type | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  |                                                                                                                           Download                                                                                                                            |
| :--------: | :-------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DeepFillv2 | free-form |  256x256   |    100k     | Places365-val |  8.635   | 22.398 | 0.815 | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_places_20200619-10d15793.log.json) |


### CelebA-HQ
|   Method   | Mask Type | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  |                                                                                                                           Download                                                                                                                            |
| :--------: | :-------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DeepFillv2 | free-form |  256x256   |     20k     | CelebA-val |  5.411   | 25.721 | 0.871 | [model](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/deepfillv2/deepfillv2_256x256_8x2_celeba_20200619-c96e5f12.log.json) |
