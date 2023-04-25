# DIM (CVPR'2017)

> [Deep Image Matting](https://arxiv.org/abs/1703.03872)

> **Task**: Matting

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Image matting is a fundamental computer vision problem and has many applications. Previous algorithms have poor performance when an image has similar foreground and background colors or complicated textures. The main reasons are prior methods 1) only use low-level features and 2) lack high-level context. In this paper, we propose a novel deep learning based algorithm that can tackle both these problems. Our deep model has two parts. The first part is a deep convolutional encoder-decoder network that takes an image and the corresponding trimap as inputs and predict the alpha matte of the image. The second part is a small convolutional network that refines the alpha matte predictions of the first network to have more accurate alpha values and sharper edges. In addition, we also create a large-scale image matting dataset including 49300 training images and 1000 testing images. We evaluate our algorithm on the image matting benchmark, our testing set, and a wide variety of real images. Experimental results clearly demonstrate the superiority of our algorithm over previous methods.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144175771-05b4d8f5-1abc-48ee-a5f1-8cc89a156e27.png" width="400"/>
</div >

## Results and models

|                                 Model                                  |    Dataset     | SAD  |  MSE  |   GRAD   |   CONN   | Training Resources |                                 Download                                  |
| :--------------------------------------------------------------------: | :------------: | :--: | :---: | :------: | :------: | :----------------: | :-----------------------------------------------------------------------: |
|         [stage1 (our)](./dim_stage1-v16_1xb1-1000k_comp1k.py)          | Composition-1k | 53.8 | 0.017 |   32.7   |   54.5   |         1          | [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage1_v16_1x1_1000k_comp1k_SAD-53.8_20200605_140257-979a420f.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage1_v16_1x1_1000k_comp1k_20200605_140257.log.json) |
|       [stage2 (our)](./dim_stage2-v16-pln_1xb1-1000k_comp1k.py)        | Composition-1k | 52.3 | 0.016 |   29.4   |   52.4   |         1          | [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage2_v16_pln_1x1_1000k_comp1k_SAD-52.3_20200607_171909-d83c4775.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage2_v16_pln_1x1_1000k_comp1k_20200607_171909.log.json) |
|       [stage3 (our)](./dim_stage3-v16-pln_1xb1-1000k_comp1k.py)        | Composition-1k | 50.6 | 0.015 | **29.0** | **50.7** |         1          | [model](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_20200609_111851.log.json) |
| [stage1 (online merge)](./dim_stage1-v16_1xb1-1000k_comp1k_online-merge.py) | Composition-1k |  -   |   -   |    -     |    -     |         -          |                                     -                                     |

<!-- |     stage1 (paper)         | Composition-1k |  54.6   |   0.017   |   36.7   |   55.3   |         -          |                                      -                                       |
|                              stage3 (paper)   |Composition-1k | **50.4** | **0.014** |   31.0   |   50.8   |         -          |                                      -                                       | -->

**NOTE**

- stage1: train the encoder-decoder part without the refinement part.
- stage2: fix the encoder-decoder part and train the refinement part.
- stage3: fine-tune the whole network.

> The performance of the model is not stable during the training. Thus, the reported performance is not from the last checkpoint. Instead, it is the best performance of all validations during training.

> The performance of training (best performance) with different random seeds diverges in a large range. You may need to run several experiments for each setting to obtain the above performance.

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

DIM is trained with three stages.

**Stage 1**: train the encoder-decoder part without the refinement part.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/dim/dim_stage1-v16_1xb1-1000k_comp1k.py

# single-gpu train
python tools/train.py configs/dim/dim_stage1-v16_1xb1-1000k_comp1k.py

# multi-gpu train
./tools/dist_train.sh configs/dim/dim_stage1-v16_1xb1-1000k_comp1k.py 8
```

**Stage 2**: fix the encoder-decoder part and train the refinement part.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/dim/dim_stage2-v16-pln_1xb1-1000k_comp1k.py

# single-gpu train
python tools/train.py configs/dim/dim_stage2-v16-pln_1xb1-1000k_comp1k.py

# multi-gpu train
./tools/dist_train.sh configs/dim/dim_stage2-v16-pln_1xb1-1000k_comp1k.py 8
```

**Stage 3**: fine-tune the whole network.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py

# single-gpu train
python tools/train.py configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py

# multi-gpu train
./tools/dist_train.sh configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth

# single-gpu test
python tools/test.py configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth

# multi-gpu test
./tools/dist_test.sh configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py https://download.openmmlab.com/mmediting/mattors/dim/dim_stage3_v16_pln_1x1_1000k_comp1k_SAD-50.6_20200609_111851-647f24b6.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@inproceedings{xu2017deep,
  title={Deep image matting},
  author={Xu, Ning and Price, Brian and Cohen, Scott and Huang, Thomas},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2970--2979},
  year={2017}
}
```
