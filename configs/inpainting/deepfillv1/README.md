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
|   Method   |  Mask Type  | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  |          Download          |
| :--------: | :---------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :------------------------: |
| DeepFillv1 | square bbox |  256x256   |    500k     | Places365-val |  11.019  | 23.429 | 0.862 | [model](xxx) \| [log](xxx) |


### CelebA-HQ
|   Method   |  Mask Type  | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  |          Download          |
| :--------: | :---------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :------------------------: |
| DeepFillv1 | square bbox |  256x256   |    150k     | CelebA-val |  6.677   | 26.878 | 0.911 | [model](xxx) \| [log](xxx) |
