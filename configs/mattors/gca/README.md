# GCA (AAAI'2020)

> [Natural Image Matting via Guided Contextual Attention](https://arxiv.org/abs/2001.04069)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Over the last few years, deep learning based approaches have achieved outstanding improvements in natural image matting. Many of these methods can generate visually plausible alpha estimations, but typically yield blurry structures or textures in the semitransparent area. This is due to the local ambiguity of transparent objects. One possible solution is to leverage the far-surrounding information to estimate the local opacity. Traditional affinity-based methods often suffer from the high computational complexity, which are not suitable for high resolution alpha estimation. Inspired by affinity-based method and the successes of contextual attention in inpainting, we develop a novel end-to-end approach for natural image matting with a guided contextual attention module, which is specifically designed for image matting. Guided contextual attention module directly propagates high-level opacity information globally based on the learned low-level affinity. The proposed method can mimic information flow of affinity-based methods and utilize rich features learned by deep neural networks simultaneously. Experiment results on Composition-1k testing set and this http URL benchmark dataset demonstrate that our method outperforms state-of-the-art approaches in natural image matting.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144176004-c9c26201-f8af-416a-9bea-ccd60bae7913.png" width="400"/>
</div >

## Results and models

|                                 Method                                  |    SAD    |    MSE     |   GRAD    |   CONN    |                                           Download                                           |
| :---------------------------------------------------------------------: | :-------: | :--------: | :-------: | :-------: | :------------------------------------------------------------------------------------------: |
|                            baseline (paper)                             |   40.62   |   0.0106   |   21.53   |   38.43   |                                              -                                               |
|                               GCA (paper)                               |   35.28   |   0.0091   |   16.92   |   32.53   |                                              -                                               |
| [baseline (our)](/configs/mattors/gca/baseline_r34_4x10_200k_comp1k.py) |   36.50   |   0.0090   |   17.40   |   34.33   | [model](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-36.50_20200614_105701-95be1750.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_20200614_105701.log.json) |
|      [GCA (our)](/configs/mattors/gca/gca_r34_4x10_200k_comp1k.py)      | **34.77** | **0.0080** | **16.33** | **32.20** | [model](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-34.77_20200604_213848-4369bea0.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_20200604_213848.log.json) |

**More results**

|                                          Method                                          |  SAD  |  MSE   | GRAD  | CONN  |                                          Download                                           |
| :--------------------------------------------------------------------------------------: | :---: | :----: | :---: | :---: | :-----------------------------------------------------------------------------------------: |
| [baseline (with DIM pipeline)](/configs/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k.py) | 49.95 | 0.0144 | 30.21 | 49.67 | [model](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_SAD-49.95_20200626_231612-535c9a11.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_20200626_231612.log.json) |
|    [GCA (with DIM pipeline)](/configs/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k.py)    | 49.42 | 0.0129 | 28.07 | 49.47 | [model](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_SAD-49.42_20200626_231422-8e9cc127.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_20200626_231422.log.json) |

## Citation

```bibtex
@inproceedings{li2020natural,
  title={Natural Image Matting via Guided Contextual Attention},
  author={Li, Yaoyi and Lu, Hongtao},
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2020}
}
```
