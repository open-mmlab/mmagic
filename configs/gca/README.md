# GCA (AAAI'2020)

> [Natural Image Matting via Guided Contextual Attention](https://arxiv.org/abs/2001.04069)

> **Task**: Matting

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Over the last few years, deep learning based approaches have achieved outstanding improvements in natural image matting. Many of these methods can generate visually plausible alpha estimations, but typically yield blurry structures or textures in the semitransparent area. This is due to the local ambiguity of transparent objects. One possible solution is to leverage the far-surrounding information to estimate the local opacity. Traditional affinity-based methods often suffer from the high computational complexity, which are not suitable for high resolution alpha estimation. Inspired by affinity-based method and the successes of contextual attention in inpainting, we develop a novel end-to-end approach for natural image matting with a guided contextual attention module, which is specifically designed for image matting. Guided contextual attention module directly propagates high-level opacity information globally based on the learned low-level affinity. The proposed method can mimic information flow of affinity-based methods and utilize rich features learned by deep neural networks simultaneously. Experiment results on Composition-1k testing set and this http URL benchmark dataset demonstrate that our method outperforms state-of-the-art approaches in natural image matting.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144176004-c9c26201-f8af-416a-9bea-ccd60bae7913.png" width="400"/>
</div >

## Results and models

|                              Model                               |    Dataset     |    SAD    |    MSE     |   GRAD    |   CONN    | Training Resources |                              Download                               |
| :--------------------------------------------------------------: | :------------: | :-------: | :--------: | :-------: | :-------: | :----------------: | :-----------------------------------------------------------------: |
|      [baseline (our)](./baseline_r34_4xb10-200k_comp1k.py)       | Composition-1K |   34.61   |   0.0083   |   16.21   |   32.12   |         4          | [model](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-34.61_20220620-96f85d56.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/baseline_r34_4x10_200k_comp1k_SAD-34.61_20220620-96f85d56.log) |
|           [GCA (our)](./gca_r34_4xb10-200k_comp1k.py)            | Composition-1K | **33.38** | **0.0081** | **14.96** | **30.59** |         4          | [model](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.log) |
| [baseline (with DIM pipeline)](./baseline_r34_4xb10-dimaug-200k_comp1k.py) | Composition-1K |   49.95   |   0.0144   |   30.21   |   49.67   |         4          | [model](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_SAD-49.95_20200626_231612-535c9a11.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/baseline_dimaug_r34_4x10_200k_comp1k_20200626_231612.log.json) |
| [GCA (with DIM pipeline)](./gca_r34_4xb10-dimaug-200k_comp1k.py) | Composition-1K |   49.42   |   0.0129   |   28.07   |   49.47   |         4          | [model](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_SAD-49.42_20200626_231422-8e9cc127.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/gca/gca_dimaug_r34_4x10_200k_comp1k_20200626_231422.log.json) |

<!-- |                   baseline (paper)                    |   40.62   |   0.0106   |   21.53   |   38.43   |         -          |                                              -                                               |
|                      GCA (paper)                      |   35.28   |   0.0091   |   16.92   |   32.53   |         -          |                                              -                                               | -->

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/gca/gca_r34_4xb10-200k_comp1k.py

# single-gpu train
python tools/train.py configs/gca/gca_r34_4xb10-200k_comp1k.py

# multi-gpu train
./tools/dist_train.sh configs/gca/gca_r34_4xb10-200k_comp1k.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/gca/gca_r34_4xb10-200k_comp1k.py https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth

# single-gpu test
python tools/test.py configs/gca/gca_r34_4xb10-200k_comp1k.py https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth

# multi-gpu test
./tools/dist_test.sh configs/gca/gca_r34_4xb10-200k_comp1k.py https://download.openmmlab.com/mmediting/mattors/gca/gca_r34_4x10_200k_comp1k_SAD-33.38_20220615-65595f39.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@inproceedings{li2020natural,
  title={Natural Image Matting via Guided Contextual Attention},
  author={Li, Yaoyi and Lu, Hongtao},
  booktitle={Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2020}
}
```
