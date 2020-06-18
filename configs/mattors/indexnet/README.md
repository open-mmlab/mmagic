# Natural Image Matting via Guided Contextual Attention

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

|   Method   |  SAD  |  MSE   | GRAD  | CONN  | Download |
|:----------:|:-----:|:------:|:-----:|:-----:|:--------:|
|  M2O DINs (paper) | **45.8**  | **0.013**  | 25.9  | **43.7**  | - |
|  M2O DINs (our)   | 46.8  | 0.016  | **24.6**  | 44.6  | [model](TODO) \| [log](TODO) |

> It should be noted that the best result we get from the original [IndexNet repo](https://github.com/poppinace/indexnet_matting) is `SAD: 46.96, MSE: 0.0143, Grad: 29.57, Conn: 46.39`
