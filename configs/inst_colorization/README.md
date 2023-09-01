# Instance-aware Image Colorization (CVPR'2020)

> [Instance-Aware Image Colorization](https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Instance-Aware_Image_Colorization_CVPR_2020_paper.html)

> **Task**: Colorization

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Image colorization is inherently an ill-posed problem with multi-modal uncertainty. Previous methods leverage the deep neural network to map input grayscale images to plausible color outputs directly. Although these learning-based methods have shown impressive performance, they usually fail on the input images that contain multiple objects. The leading cause is that existing models perform learning and colorization on the entire image. In the absence of a clear figure-ground separation, these models cannot effectively locate and learn meaningful object-level semantics. In this paper, we propose a method for achieving instance-aware colorization. Our network architecture leverages an off-the-shelf object detector to obtain cropped object images and uses an instance colorization network to extract object-level features. We use a similar network to extract the full-image features and apply a fusion module to full object-level and image-level features to predict the final colors. Both colorization networks and fusion modules are learned from a large-scale dataset. Experimental results show that our work outperforms existing methods on different quality metrics and achieves state-of-the-art performance on image colorization.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://github.com/ericsujw/InstColorization/blob/master/imgs/teaser.png?raw=true" width="400"/>
</div >

## Results and models

|                                              Model                                              | Dataset |                                              Download                                              |
| :---------------------------------------------------------------------------------------------: | :-----: | :------------------------------------------------------------------------------------------------: |
| [instance_aware_colorization_officiial](./inst-colorizatioon_full_official_cocostuff-256x256.py) | MS-COCO | [model](https://download.openmmlab.com/mmediting/inst_colorization/inst-colorizatioon_full_official_cocostuff-256x256-5b9d4eee.pth) |

## Quick Start

<details>
<summary>Colorization demo</summary>

You can use the following commands to colorize an image.

```shell

python demo/mmagic_inference_demo.py --model-name inst_colorization --img input.jpg --result-out-dir output.png
```

For more demos, you can refer to [Tutorial 3: inference with pre-trained models](https://mmagic.readthedocs.io/en/latest/user_guides/3_inference.html).

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
