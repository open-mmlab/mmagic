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
|    Method    |  Mask Type  | Train Iters |   Test Set    | l1 error | l2 error |   PSNR   |   SSIM   |   TV   |            Download            |
| :----------: | :---------: | :---------: | :-----------: | :------: | :------: | :------: | :------: | :----: | :----------------------------: |
| Global&Local | square bbox |    500k     | Places365-val |  l1:xxx  |  l2:xxx  | psnr:xxx | ssim:xxx | tv:xxx | [model](xxx) &#124; [log](xxx) |


### CelebA-HQ
|    Method    |  Mask Type  | Train Iters |  Test Set  | l1 error | l2 error |   PSNR   |   SSIM   |   TV   |            Download            |
| :----------: | :---------: | :---------: | :--------: | :------: | :------: | :------: | :------: | :----: | :----------------------------: |
| Global&Local | square bbox |    500k     | CelebA-val |  l1:xxx  |  l2:xxx  | psnr:xxx | ssim:xxx | tv:xxx | [model](xxx) &#124; [log](xxx) |
