# RealBasicVSR (CVPR'2022)

> [RealBasicVSR: Investigating Tradeoffs in Real-World Video Super-Resolution](https://arxiv.org/abs/2111.12704)

> **Task**: Video Super-Resolution

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The diversity and complexity of degradations in real-world video super-resolution (VSR) pose non-trivial challenges in inference and training. First, while long-term propagation leads to improved performance in cases of mild degradations, severe in-the-wild degradations could be exaggerated through propagation, impairing output quality. To balance the tradeoff between detail synthesis and artifact suppression, we found an image pre-cleaning stage indispensable to reduce noises and artifacts prior to propagation. Equipped with a carefully designed cleaning module, our RealBasicVSR outperforms existing methods in both quality and efficiency. Second, real-world VSR models are often trained with diverse degradations to improve generalizability, requiring increased batch size to produce a stable gradient. Inevitably, the increased computational burden results in various problems, including 1) speed-performance tradeoff and 2) batch-length tradeoff. To alleviate the first tradeoff, we propose a stochastic degradation scheme that reduces up to 40% of training time without sacrificing performance. We then analyze different training settings and suggest that employing longer sequences rather than larger batches during training allows more effective uses of temporal information, leading to more stable performance during inference. To facilitate fair comparisons, we propose the new VideoLQ dataset, which contains a large variety of real-world low-quality video sequences containing rich textures and patterns. Our dataset can serve as a common ground for benchmarking. Code, models, and the dataset will be made publicly available.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/7676947/146704029-58bc4db4-267f-4158-8129-e49ab6652249.png" width="400"/>
</div >

## Results and models

Evaluated on Y channel. The code for computing NRQM, NIQE, and PI can be found [here](https://github.com/roimehrez/PIRM2018). MATLAB official code is used to compute BRISQUE.

|                               Model                                | Dataset | NRQM (Y) | NIQE (Y) | PI (Y) | BRISQUE (Y) |    Training Resources    |                                Download                                |
| :----------------------------------------------------------------: | :-----: | :------: | :------: | :----: | :---------: | :----------------------: | :--------------------------------------------------------------------: |
| [realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds](./realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py) |  REDS   |  6.0477  |  3.7662  | 3.8593 |   29.030    | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth)/[log](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104_183640.log.json) |
| [realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds](./realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py) |  REDS   |    -     |    -     |   -    |      -      | 8 (Tesla V100-SXM2-32GB) | [model](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds_20211027-0e2ff207.pth)/[log](http://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds_20211027_114039.log.json) |

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py

# single-gpu train
python tools/train.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py

# multi-gpu train
./tools/dist_train.sh configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth

# single-gpu test
python tools/test.py python tools/test.py configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth

# multi-gpu test
./tools/dist_test.sh configs/real_basicvsr/realbasicvsr_c64b20-1x30x8_8xb1-lr5e-5-150k_reds.py https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@InProceedings{chan2022investigating,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {RealBasicVSR: Investigating Tradeoffs in Real-World Video Super-Resolution},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2022}
}
```
