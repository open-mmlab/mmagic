# Benchmark and Model Zoo for Matting

## Benchmark

### Deep Image Matting (DIM)

Please refer to [DIM](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/dim) for details.

### GCA Matting

Please refer to [GCA](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/gca) for details.

### IndexNet Matting

Please refer to [IndexNet](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/indexnet) for details.


## Overview

|        Method       |  SAD  |   MSE  |  GRAD |  CONN |
|:-------------------:|:-----:|:------:|:-----:|:-----:|
|        DIM          | 50.6  | 0.015  | 29.0  | 50.7  |
|        GCA          | 34.77 | 0.0080 | 16.33 | 32.20 |
|      IndexNet       | 46.8  | 0.016  | 24.6  | 44.6  |


## Evaluation Details

### Data

We provide a python script [preprocess_comp1k_dataset.py](https://github.com/open-mmlab/mmediting/blob/master/tools/preprocess_comp1k_dataset.py) for compositing Adobe Composition-1k (comp1k) dataset foreground images with MS COCO dataset background images. The result merged images are the same as the merged images produced by the official composite script by Adobe.

### Evaluation Implementation Details

We provide a python script [evaluate_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/tools/evaluate_comp1k.py) for evaluating test results of matting models. The four evaluation metrics (`SAD`, `MSE`, `GRAD` and `CONN`) are calculated in the same way as the official evaluation script by Adobe. We observe only minor difference between the evaluation results of our python script and the official, which has no effect on the reported performance.
