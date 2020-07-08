# Natural Image Matting via Guided Contextual Attention

## Introduction

```
@inproceedings{li2020natural,
  title={Natural Image Matting via Guided Contextual Attention},
  author={Li, Yaoyi and Lu, Hongtao},
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2020}
}
```

## Results and Models

|   Method   |  SAD  |  MSE   | GRAD  | CONN  | Download |
|:----------:|:-----:|:------:|:-----:|:-----:|:--------:|
| baseline (paper) | 40.62 | 0.0106 | 21.53 | 38.43 | -  |
|   GCA (paper)    | 35.28 | 0.0091 | 16.92 | 32.53 | -  |
|  baseline (our)  | 36.50 | 0.0090 | 17.40 | 34.33 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmediting/v0.1/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-36.50_20200614_105701-95be1750.pth) \| [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmediting/v0.1/mattors/gca/baseline_r34_4x10_200k_comp1k_20200614_105701.log.json) |
|    GCA (our)     | **34.77** | **0.0080** | **16.33** | **32.20** | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmediting/v0.1/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-34.77_20200604_213848-4369bea0.pth) \| [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmediting/v0.1/mattors/gca/gca_r34_4x10_200k_comp1k_20200604_213848.log.json) |

### More results

|   Method   |  SAD  |  MSE   | GRAD  | CONN  | Download |
|:----------:|:-----:|:------:|:-----:|:-----:|:--------:|
| baseline (with DIM pipeline) | 49.95 | 0.0144 | 30.21 | 49.67 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmediting/v0.1/mattors/gca/TODO_to_be_added) \| [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmediting/v0.1/mattors/gca/TODO_to_be_added) |
|    GCA (with DIM pipeline)   | 49.42 | 0.0129 | 28.07 | 49.47 | [model](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmediting/v0.1/mattors/gca/TODO_to_be_added) \| [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmediting/v0.1/mattors/gca/TODO_to_be_added) |
