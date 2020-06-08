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
| Method | Mask Type | Resolution | Train Iters |   Test Set    | l1 error | l2 error |   PSNR   |   SSIM   |   TV   |            Download            |
| :----: | :-------: | :--------: | :---------: | :-----------: | :------: | :------: | :------: | :------: | :----: | :----------------------------: |
| PConv  | free-form |  256x256   |    500k     | Places365-val |  l1:xxx  |  l2:xxx  | psnr:xxx | ssim:xxx | tv:xxx | [model](xxx) &#124; [log](xxx) |


### CelebA-HQ
| Method | Mask Type | Resolution | Train Iters |  Test Set  | l1 error | l2 error |   PSNR   |   SSIM   |   TV   |            Download            |
| :----: | :-------: | :--------: | :---------: | :--------: | :------: | :------: | :------: | :------: | :----: | :----------------------------: |
| PConv  | free-form |  256x256   |    500k     | CelebA-val |  l1:xxx  |  l2:xxx  | psnr:xxx | ssim:xxx | tv:xxx | [model](xxx) &#124; [log](xxx) |
