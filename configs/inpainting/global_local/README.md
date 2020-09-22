# Globally and Locally Consistent Image Completion

## Introduction

```
@article{iizuka2017globally,
  title={Globally and locally consistent image completion},
  author={Iizuka, Satoshi and Simo-Serra, Edgar and Ishikawa, Hiroshi},
  journal={ACM Transactions on Graphics (ToG)},
  volume={36},
  number={4},
  pages={1--14},
  year={2017},
  publisher={ACM New York, NY, USA}
}
```

*Note that we do not apply the post-processing module in Global&Local for a fair comparison with current deep inpainting methods.*

## Results and models
### Places365-Challenge
|    Method    |  Mask Type  | Resolution | Train Iters |   Test Set    | l1 error |  PSNR  | SSIM  |                                                                                                                      Download                                                                                                                       |
| :----------: | :---------: | :--------: | :---------: | :-----------: | :------: | :----: | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Global&Local | square bbox |  256x256   |    500k     | Places365-val |  11.164  | 23.152 | 0.862 | [model](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_places_20200619-52a040a8.log.json) |


### CelebA-HQ
|    Method    |  Mask Type  | Resolution | Train Iters |  Test Set  | l1 error |  PSNR  | SSIM  |                                                                                                                      Download                                                                                                                       |
| :----------: | :---------: | :--------: | :---------: | :--------: | :------: | :----: | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Global&Local | square bbox |  256x256   |    500k     | CelebA-val |  6.678   | 26.780 | 0.904 | [model](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth) \| [log](https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.log.json) |
