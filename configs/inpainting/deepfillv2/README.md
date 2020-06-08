# Generative Image Inpainting with Contextual Attention

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
|   Method   | Mask Type | Resolution | Train Iters |   Test Set    | l1 error | l2 error |   PSNR   |   SSIM   |   TV   |            Download            |
| :--------: | :-------: | :--------: | :---------: | :-----------: | :------: | :------: | :------: | :------: | :----: | :----------------------------: |
| DeepFillv2 | free-form |  256x256   |    500k     | Places365-val |  l1:xxx  |  l2:xxx  | psnr:xxx | ssim:xxx | tv:xxx | [model](xxx) &#124; [log](xxx) |


### CelebA-HQ
|   Method   | Mask Type | Resolution | Train Iters |  Test Set  | l1 error | l2 error |   PSNR   |   SSIM   |   TV   |            Download            |
| :--------: | :-------: | :--------: | :---------: | :--------: | :------: | :------: | :------: | :------: | :----: | :----------------------------: |
| DeepFillv2 | free-form |  256x256   |    500k     | CelebA-val |  l1:xxx  |  l2:xxx  | psnr:xxx | ssim:xxx | tv:xxx | [model](xxx) &#124; [log](xxx) |
