# pix2pix: Image-To-Image Translation With Conditional Adversarial Networks

## Introduction

```
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```

## Results and Models

We use `FID` and `IS` metrics to 	evaluate the generation performance of pix2pix.

`FID` evaluation:

| Dataset | facades | maps-a2b | maps-b2a | edges2shoes | average |
|:---:|:---:|:---:|:---:|:---:|:---:|
| official | **119.135** | 149.731 | 102.072 | **75.774** | 111.678 |
| ours | 127.792 | **118.552** | **92.798** | 85.413 | **106.139** |

`IS` evaluation:

| Dataset | facades | maps-a2b | maps-b2a | edges2shoes | average |
|:---:|:---:|:---:|:---:|:---:|:---:|
| official | 1.650 | 2.529 | **3.552** | **2.766** | 2.624 |
| ours | **1.745** | **2.689** | 3.473 | 2.747 | **2.664** |

Model and log downloads:

| Dataset | facades | maps-a2b | maps-b2a | edges2shoes |
|:---:|:---:|:---:|:---:|:---:|
| download | [model](TODO) \| [log](TODO) | [model](TODO) \| [log](TODO) | [model](TODO) \| [log](TODO) | [model](TODO) \| [log](TODO) |

Note: we strictly follow the [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf) setting in Section 3.3: "*At inference time, we run the generator net in exactly
the same manner as during the training phase. This differs
from the usual protocol in that we apply dropout at test time,
and we apply batch normalization using the statistics of
the test batch, rather than aggregated statistics of the training batch.*" (i.e., use model.train() mode), thus may lead to slightly different inference results every time.
