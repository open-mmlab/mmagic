# GCA (AAAI'2020)

<!-- [ALGORITHM] -->

<details>
<summary align="right">GCA (AAAI'2020)</summary>

```bibtex
@inproceedings{li2020natural,
  title={Natural Image Matting via Guided Contextual Attention},
  author={Li, Yaoyi and Lu, Hongtao},
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2020}
}
```

</details>

<br/>

|                             算法                              |    SAD    |    MSE     |   GRAD    |   CONN    | GPU 信息 |                                                                                                                           下载                                                                                                                           |
| :-----------------------------------------------------------: | :-------: | :--------: | :-------: | :-------: | :------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                          基线 (原文)                          |   40.62   |   0.0106   |   21.53   |   38.43   |    -     |                                                                                                                            -                                                                                                                             |
|                          GCA (原文)                           |   35.28   |   0.0091   |   16.92   |   32.53   |    -     |                                                                                                                            -                                                                                                                             |
| [基线 (复现)](/configs/gca/baseline_r34_200k-4xb10_comp1k.py) |   34.61   |   0.0083   |   16.21   |   32.12   |    4     | [模型](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-34.61_20220620-96f85d56.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-34.61_20220620-96f85d56.log) |
|    [GCA (复现)](/configs/gca/gca_r34_200k-4xb10_comp1k.py)    | **33.38** | **0.0081** | **14.96** | **30.59** |    4     |      [模型](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.log)      |

**其他结果**

|                                      算法                                       |  SAD  |  MSE   | GRAD  | CONN  | GPU 信息 |                                                                                                                                  下载                                                                                                                                  |
| :-----------------------------------------------------------------------------: | :---: | :----: | :---: | :---: | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [基线 (使用 DIM 流水线)](/configs/gca/baseline_r34_dimaug-200k-4xb10_comp1k.py) | 49.95 | 0.0144 | 30.21 | 49.67 |    4     | [模型](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_SAD-49.95_20200626_231612-535c9a11.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_20200626_231612.log.json) |
|    [GCA (使用 DIM 流水线)](/configs/gca/gca_r34_dimaug-200k-4xb10_comp1k.py)    | 49.42 | 0.0129 | 28.07 | 49.47 |    4     |      [模型](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_SAD-49.42_20200626_231422-8e9cc127.pth) \| [日志](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_20200626_231422.log.json)      |
