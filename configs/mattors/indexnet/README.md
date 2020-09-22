# Indices Matter: Learning to Index for Deep Image Matting

## Introduction

```
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## Results and Models

|      Method      |   SAD    |    MSE    |   GRAD   |   CONN   |                                                                                                                              Download                                                                                                                               |
| :--------------: | :------: | :-------: | :------: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| M2O DINs (paper) |   45.8   |   0.013   |   25.9   | **43.7** |                                                                                                                                  -                                                                                                                                  |
|  M2O DINs (our)  | **45.6** | **0.012** | **25.5** |   44.8   | [model](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_20200618_173817.log.json) |

> The performance of training (best performance) with different random seeds diverges in a large range. You may need to run several experiments for each setting to obtain the above performance.

### More result

|            Method            |  SAD  |  MSE  | GRAD  | CONN  |                                                                                                                                     Download                                                                                                                                      |
| :--------------------------: | :---: | :---: | :---: | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| M2O DINs (with DIM pipeline) | 50.1  | 0.016 | 30.8  | 49.5  | [model](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k_SAD-50.1_20200626_231857-af359436.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k_20200626_231857.log.json) |
