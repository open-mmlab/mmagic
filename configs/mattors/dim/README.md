# Deep Image Matting

## Introduction

```
@inproceedings{xu2017deep,
  title={Deep image matting},
  author={Xu, Ning and Price, Brian and Cohen, Scott and Huang, Thomas},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2970--2979},
  year={2017}
}
```

## Results and Models

|     Method     |   SAD    |    MSE    |   GRAD   |   CONN   |                                                                                                                              Download                                                                                                                               |
| :------------: | :------: | :-------: | :------: | :------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| stage1 (paper) |   54.6   |   0.017   |   36.7   |   55.3   |                                                                                                                                  -                                                                                                                                  |
| stage3 (paper) | **50.4** | **0.014** |   31.0   |   50.8   |                                                                                                                                  -                                                                                                                                  |
|  stage1 (our)  |   53.8   |   0.017   |   32.7   |   54.5   |     [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage1_v16_1x1_1000k_comp1k_SAD-53.8_20200605_140257-979a420f.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage1_v16_1x1_1000k_comp1k_20200605_140257.log.json)     |
|  stage2 (our)  |   52.3   |   0.016   |   29.4   |   52.4   | [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage2_v16_pln_1x1_1000k_comp1k_20200607_171909.log.json) |
|  stage3 (our)  |   50.6   |   0.015   | **29.0** | **50.7** | [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_20200609_111851.log.json) |

**NOTE**

* stage1: train the encoder-decoder part without the refinement part. \
* stage2: fix the encoder-decoder part and train the refinement part. \
* stage3: fine-tune the whole network.

> The performance of the model is not stable during the training. Thus, the reported performance is not from the last checkpoint. Instead, it is the best performance of all validations during training.

> The performance of training (best performance) with different random seeds diverges in a large range. You may need to run several experiments for each setting to obtain the above performance.
