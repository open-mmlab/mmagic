# Instance-aware Image Colorization (CVPR'2020)

> [Instance-Aware Image Colorization](https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Instance-Aware_Image_Colorization_CVPR_2020_paper.html)

> **Task**: Colorization

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Image colorization is inherently an ill-posed problem with multi-modal uncertainty. Previous methods leverage the deep neural network to map input grayscale images to plausible color outputs directly. Although these learning-based methods have shown impressive performance, they usually fail on the input images that contain multiple objects. The leading cause is that existing models perform learning and colorization on the entire image. In the absence of a clear figure-ground separation, these models cannot effectively locate and learn meaningful object-level semantics. In this paper, we propose a method for achieving instance-aware colorization. Our network architecture leverages an off-the-shelf object detector to obtain cropped object images and uses an instance colorization network to extract object-level features. We use a similar network to extract the full-image features and apply a fusion module to full object-level and image-level features to predict the final colors. Both colorization networks and fusion modules are learned from a large-scale dataset. Experimental results show that our work outperforms existing methods on different quality metrics and achieves state-of-the-art performance on image colorization.

## Results and models

## Quick Start

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# CPU train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/insta/insta_full_cocostuff_256x256.py

# single-gpu train
python tools/train.py configs/insta/insta_full_cocostuff_256x256.py

# multi-gpu train
./tools/dist_train.sh configs/insta/insta_full_cocostuff_256x256.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# CPU test
CUDA_VISIBLE_DEVICES=-1 python demo/colorization_demo.py configs/insta/insta_full_cocostuff_256x256.py ../checkpoints/instance_aware_cocostuff.pth

# single-gpu test
python demo/colorization_demo.py configs/insta/insta_full_cocostuff_256x256.py ../checkpoints/instance_aware_cocostuff.pth

# multi-gpu test
./tools/dist_test.sh configs/insta/insta_full_cocostuff_256x256.py ../checkpoints/instance_aware_cocostuff.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>

<details>
<summary align="right">Instance-aware Image Colorization (CVPR'2020)</summary>

```bibtex
@inproceedings{Su-CVPR-2020,
  author = {Su, Jheng-Wei and Chu, Hung-Kuo and Huang, Jia-Bin},
  title = {Instance-aware Image Colorization},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

</details>
