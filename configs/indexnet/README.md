# IndexNet (ICCV'2019)

> [Indices Matter: Learning to Index for Deep Image Matting](https://arxiv.org/abs/1908.00672)

> **Task**: Matting

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We show that existing upsampling operators can be unified with the notion of the index function. This notion is inspired by an observation in the decoding process of deep image matting where indices-guided unpooling can recover boundary details much better than other upsampling operators such as bilinear interpolation. By looking at the indices as a function of the feature map, we introduce the concept of learning to index, and present a novel index-guided encoder-decoder framework where indices are self-learned adaptively from data and are used to guide the pooling and upsampling operators, without the need of supervision. At the core of this framework is a flexible network module, termed IndexNet, which dynamically predicts indices given an input. Due to its flexibility, IndexNet can be used as a plug-in applying to any off-the-shelf convolutional networks that have coupled downsampling and upsampling stages.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/12726765/144176083-52604501-1f46-411d-b81a-cad0eb4b529f.png" width="400"/>
</div >

## Results and models

|                                Model                                 |    Dataset     |   SAD    |    MSE    |   GRAD   | CONN | Training Resources |                                Download                                 |
| :------------------------------------------------------------------: | :------------: | :------: | :-------: | :------: | :--: | :----------------: | :---------------------------------------------------------------------: |
|        [M2O DINs (our)](./indexnet_mobv2_1xb16-78k_comp1k.py)        | Composition-1k | **45.6** | **0.012** | **25.5** | 44.8 |         1          | [model](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_20200618_173817.log.json) |
| [M2O DINs (with DIM pipeline)](./indexnet_mobv2-dimaug_1xb16-78k_comp1k.py) | Composition-1k |   50.1   |   0.016   |   30.8   | 49.5 |         1          | [model](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k_SAD-50.1_20200626_231857-af359436.pth) \| [log](https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_dimaug_mobv2_1x16_78k_comp1k_20200626_231857.log.json) |

<!-- |                    M2O DINs (paper)                    |   45.8   |   0.013   |   25.9   | **43.7** |         -          |                     -->

> The performance of training (best performance) with different random seeds diverges in a large range. You may need to run several experiments for each setting to obtain the above performance.

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py

# single-gpu train
python tools/train.py configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py

# multi-gpu train
./tools/dist_train.sh configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMagic).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth

# single-gpu test
python tools/test.py configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth

# multi-gpu test
./tools/dist_test.sh configs/indexnet/indexnet_mobv2_1xb16-78k_comp1k.py https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@inproceedings{hao2019indexnet,
  title={Indices Matter: Learning to Index for Deep Image Matting},
  author={Lu, Hao and Dai, Yutong and Shen, Chunhua and Xu, Songcen},
  booktitle={Proc. IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
