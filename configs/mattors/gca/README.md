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

|      Method      |    SAD    |    MSE     |   GRAD    |   CONN    |                                                                                                                         Download                                                                                                                         |
| :--------------: | :-------: | :--------: | :-------: | :-------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| baseline (paper) |   40.62   |   0.0106   |   21.53   |   38.43   |                                                                                                                            -                                                                                                                             |
|   GCA (paper)    |   35.28   |   0.0091   |   16.92   |   32.53   |                                                                                                                            -                                                                                                                             |
|  baseline (our)  |   36.50   |   0.0090   |   17.40   |   34.33   | [model](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-36.50_20200614_105701-95be1750.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_20200614_105701.log.json) |
|    GCA (our)     | **34.77** | **0.0080** | **16.33** | **32.20** |      [model](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-34.77_20200604_213848-4369bea0.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_20200604_213848.log.json)      |

### More results

|            Method            |  SAD  |  MSE   | GRAD  | CONN  |                                                                                                                                Download                                                                                                                                |
| :--------------------------: | :---: | :----: | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| baseline (with DIM pipeline) | 49.95 | 0.0144 | 30.21 | 49.67 | [model](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_SAD-49.95_20200626_231612-535c9a11.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_20200626_231612.log.json) |
|   GCA (with DIM pipeline)    | 49.42 | 0.0129 | 28.07 | 49.47 |      [model](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_SAD-49.42_20200626_231422-8e9cc127.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_20200626_231422.log.json)      |
