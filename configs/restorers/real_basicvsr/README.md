# RealBasicVSR (arXiv'2021)

## Abstract

<!-- [ABSTRACT] -->

The diversity and complexity of degradations in real-world video super-resolution (VSR) pose non-trivial challenges in inference and training. First, while long-term propagation leads to improved performance in cases of mild degradations, severe in-the-wild degradations could be exaggerated through propagation, impairing output quality. To balance the tradeoff between detail synthesis and artifact suppression, we found an image pre-cleaning stage indispensable to reduce noises and artifacts prior to propagation. Equipped with a carefully designed cleaning module, our RealBasicVSR outperforms existing methods in both quality and efficiency. Second, real-world VSR models are often trained with diverse degradations to improve generalizability, requiring increased batch size to produce a stable gradient. Inevitably, the increased computational burden results in various problems, including 1) speed-performance tradeoff and 2) batch-length tradeoff. To alleviate the first tradeoff, we propose a stochastic degradation scheme that reduces up to 40% of training time without sacrificing performance. We then analyze different training settings and suggest that employing longer sequences rather than larger batches during training allows more effective uses of temporal information, leading to more stable performance during inference. To facilitate fair comparisons, we propose the new VideoLQ dataset, which contains a large variety of real-world low-quality video sequences containing rich textures and patterns. Our dataset can serve as a common ground for benchmarking. Code, models, and the dataset will be made publicly available.

<!-- [IMAGE] -->
<p align="center">
  <img src="https://user-images.githubusercontent.com/7676947/146704029-58bc4db4-267f-4158-8129-e49ab6652249.png" />
</p>

<!-- [PAPER_TITLE: Investigating Tradeoffs in Real-World Video Super-Resolution] -->
<!-- [PAPER_URL: https://arxiv.org/abs/2111.12704] -->

## Citation

<!-- [ALGORITHM] -->

```bibtex
@article{chan2021investigating,
  author = {Chan, Kelvin C.K. and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  title = {Investigating Tradeoffs in Real-World Video Super-Resolution},
  journal = {arXiv preprint arXiv:2111.12704},
  year = {2021}
}
```

## Results and models

Evaluated on Y channel. The code for computing NRQM, NIQE, and PI can be found [here](https://github.com/roimehrez/PIRM2018). MATLAB official code is used to compute BRISQUE.

|                                                             Method                                                             | NRQM (Y) | NIQE (Y) | PI (Y) | BRISQUE (Y) |                                                                                                                                         Download                                                                                                                                         |
|:------------------------------------------------------------------------------------------------------------------------------:|:--------:|:--------:|:------:|:-----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds](/configs/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py) |  6.0477  |  3.7662  | 3.8593 |    29.030   | [model](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth)/[log](https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104_183640.log.json) |



## Training
The training is divided into two stages:
1. Train a model without perceptual loss and adversarial loss using [realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds.py](realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds.py).
2. Finetune the model with perceptual loss and adversarial loss using [realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py](realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds.py).

**Note:**
1. You may want to crop the images into sub-images for faster IO. Please refer to [here](/tools/data/super-resolution/reds/preprocess_reds_dataset.py) for more details.
