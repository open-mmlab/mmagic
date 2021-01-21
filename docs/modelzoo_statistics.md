
# Model Zoo

## Model Zoo Statistics
* Number of checkpoints: 30
* Number of papers: 15
   - ALGORITHM: 15


### Inpainting

* Number of checkpoints: 8
* Number of papers: 4
    - [ALGORITHM] free-form image inpainting with gated convolution
    - [ALGORITHM] generative image inpainting with contextual attention
    - [ALGORITHM] globally and locally consistent image completion
    - [ALGORITHM] image inpainting for irregular holes using partial convolutions



### Mattors

* Number of checkpoints: 9
* Number of papers: 3
    - [ALGORITHM] deep image matting
    - [ALGORITHM] indices matter: learning to index for deep image matting
    - [ALGORITHM] natural image matting via guided contextual attention



### Restorers

* Number of checkpoints: 11
* Number of papers: 6
    - [ALGORITHM] edvr: video restoration with enhanced deformable convolutional networks
    - [ALGORITHM] enhanced deep residual networks for single image super-resolution
    - [ALGORITHM] esrgan: enhanced super-resolution generative adversarial networks
    - [ALGORITHM] image super-resolution using deep convolutional networks
    - [ALGORITHM] photo-realistic single image super-resolution using a generative adversarial network
    - [ALGORITHM] video enhancement with task-oriented flow



### Synthesizers

* Number of checkpoints: 2
* Number of papers: 2
    - [ALGORITHM] image-to-image translation with conditional adversarial networks
    - [ALGORITHM] unpaired image-to-image translation using cycle-consistent adversarial networks




## Inpainting

### Benchmark

#### Global&Local

Please refer to [GL](https://github.com/open-mmlab/mmediting/blob/master/configs/inpainting/global_local) for details.

#### Partial Conv

Please refer to [PConv](https://github.com/open-mmlab/mmediting/blob/master/configs/inpainting/partial_conv) for details.

#### DeepFillv1

Please refer to [DeepFillv1](https://github.com/open-mmlab/mmediting/blob/master/configs/inpainting/deepfillv1) for details.

#### DeepFillv2

Please refer to [DeepFillv2](https://github.com/open-mmlab/mmediting/blob/master/configs/inpainting/deepfillv2) for details.


## Matting

### Overview

|        Method       |  SAD  |   MSE  |  GRAD |  CONN |
|:-------------------:|:-----:|:------:|:-----:|:-----:|
|        DIM          | 50.62 | 0.0151 | 29.01 | 50.69 |
|        GCA          | 34.77 | 0.0080 | 16.33 | 32.20 |
|      IndexNet       | 45.56 | 0.0125 | 25.49 | 44.79 |

Above results follow the original implementation of these methods.
However, they adopt different data augmentations and preprocessing pipelines.
We also provide a benchmark for these methods under the same settings, i.e., using the same data augmentations as DIM. Results are shown as below.

|        Method       |  SAD  |   MSE  |  GRAD |  CONN |
|:-------------------:|:-----:|:------:|:-----:|:-----:|
|        DIM          | 50.62 | 0.0151 | 29.01 | 50.69 |
|        GCA*         | 49.42 | 0.0129 | 28.07 | 49.47 |
|      IndexNet*      | 50.11 | 0.0164 | 30.82 | 49.53 |

> *: We only run one experiment under the setting.

### Benchmark

#### Deep Image Matting (DIM)

Please refer to [DIM](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/dim) for details.

#### GCA Matting

Please refer to [GCA](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/gca) for details.

#### IndexNet Matting

Please refer to [IndexNet](https://github.com/open-mmlab/mmediting/blob/master/configs/mattors/indexnet) for details.

### Evaluation Details

#### Data

We provide a python script [preprocess_comp1k_dataset.py](https://github.com/open-mmlab/mmediting/blob/master/tools/preprocess_comp1k_dataset.py) for compositing Adobe Composition-1k (comp1k) dataset foreground images with MS COCO dataset background images. The result merged images are the same as the merged images produced by the official composite script by Adobe.

#### Evaluation Implementation Details

We provide a python script [evaluate_comp1k.py](https://github.com/open-mmlab/mmediting/blob/master/tools/evaluate_comp1k.py) for evaluating test results of matting models. The four evaluation metrics (`SAD`, `MSE`, `GRAD` and `CONN`) are calculated in the same way as the official evaluation script by Adobe. We observe only minor difference between the evaluation results of our python script and the official, which has no effect on the reported performance.


## Restoration

### Benchmark

#### EDSR

Please refer to [EDSR](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/edsr) for details.

#### EDVR

Please refer to [EDVR](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/edvr) for details.

#### ESRGAN

Please refer to [ESRGAN](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/esrgan) for details.

#### SRCNN

Please refer to [SRCNN](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/srcnn) for details.

#### SRResNet and SRGAN

Please refer to [SRResNet and SRGAN](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/srresnet_srgan) for details.

#### TOF

Please refer to [TOF](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/tof) for details.


## Generation

### Benchmark

#### pix2pix

Please refer to [pix2pix](https://github.com/open-mmlab/mmediting/blob/master/configs/synthesizers/pix2pix) for details.

#### CycleGAN

Please refer to [CycleGAN](https://github.com/open-mmlab/mmediting/blob/master/configs/synthesizers/cyclegan) for details.
