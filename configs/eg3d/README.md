# EG3D (CVPR'2022)

> [Efficient geometry-aware 3D generative adversarial networks](https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Efficient_Geometry-Aware_3D_Generative_Adversarial_Networks_CVPR_2022_paper.html)

> **Task**: 3D-aware Generation

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Unsupervised generation of high-quality multi-view-consistent images and 3D shapes using only collections of single-view 2D photographs has been a long-standing challenge. Existing 3D GANs are either compute-intensive or make approximations that are not 3D-consistent; the former limits quality and resolution of the generated images and the latter adversely affects multi-view consistency and shape quality. In this work, we improve the computational efficiency and image quality of 3D GANs without overly relying on these approximations. We introduce an expressive hybrid explicit-implicit network architecture that, together with other design choices, synthesizes not only high-resolution multi-view-consistent images in real time but also produces high-quality 3D geometry. By decoupling feature generation and neural rendering, our framework is able to leverage state-of-the-art 2D CNN generators, such as StyleGAN2, and inherit their efficiency and expressiveness. We demonstrate state-of-the-art 3D-aware synthesis with FFHQ and AFHQ Cats, among other experiments.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/204269503-b66a6761-00e8-49ba-842f-65aae3110278.png"/>
</div>

## Results and Models

|                            Model                            |    Dataset    |     Comment     | FID50k | FID50k-Camera |                                            Download                                            |
| :---------------------------------------------------------: | :-----------: | :-------------: | :----: | :-----------: | :--------------------------------------------------------------------------------------------: |
| [ShapeNet-Car](./eg3d_cvt-official-rgb_shapenet-128x128.py) | ShaperNet-Car | official weight | 5.6573 |    5.2325     | [model](https://download.openmmlab.com/mmediting/eg3d/eg3d_cvt-official-rgb_shapenet-128x128-85757f4d.pth) |
|       [AFHQ](./eg3d_cvt-official-rgb_afhq-512x512.py)       |     AFHQ      | official weight | 2.9134 |    6.4213     | [model](https://download.openmmlab.com/mmediting/eg3d/eg3d_cvt-official-rgb_afhq-512x512-ca1dd7c9.pth) |
|       [FFHQ](./eg3d_cvt-official-rgb_ffhq-512x512.py)       |     FFHQ      | official weight | 4.3076 |    6.4453     | [model](https://download.openmmlab.com/mmediting/eg3d/eg3d_cvt-official-rgb_ffhq-512x512-5a0ddcb6.pth) |

- `FID50k-Camera` denotes image generated with random sampled camera position.
- `FID50k` denotes image generated with camera position randomly sampled from the original dataset.

### Influence of FP16

All metrics are evaluated under FP32, and it's hard to determine how they will change if we use FP16.
For example, if we use FP16 at the super resolution module in [FFHQ model](./eg3d_cvt-official-rgb_ffhq-512x512.py), the output images will be slightly blurrier than the ones generated under FP32, but FID (**4.03**) will be better than FP32 ones.

## About generate images and videos with High-Level API

You can use the following command to generate sequence images with continuous changed camera position as input.

```shell
python demo/mmagic_inference_demo.py --model-name eg3d \
    --model-config configs/eg3d/eg3d_cvt-official-rgb_afhq-512x512.py \
    --model-ckpt https://download.openmmlab.com/mmediting/eg3d/eg3d_cvt-official-rgb_afhq-512x512-ca1dd7c9.pth \
    --result-out-dir eg3d_output \  # save images and videos to `eg3d_output`
    --interpolation camera \  # interpolation camera position only
    --num-images 100  # generate 100 images during interpolation
```

The the following video will be saved to `eg3d_output`.

<div align=center>
<video src="https://user-images.githubusercontent.com/28132635/204278664-b73b133b-9c3f-4a87-8750-133b7dedaebb.mp4"/>
</div>

To interpolate the camera position and style code at the same time, you can use the following command.

```shell
python demo/mmagic_inference_demo.py --model-name eg3d \
    --model-config configs/eg3d/eg3d_cvt-official-rgb_ffhq-512x512.py \
    --model-ckpt https://download.openmmlab.com/mmediting/eg3d/eg3d_cvt-official-rgb_ffhq-512x512-5a0ddcb6.pth \
    --result-out-dir eg3d_output \  # save images and videos to `eg3d_output`
    --interpolation both \  # interpolation camera and conditioning both
    --num-images 100  # generate 100 images during interpolation
    --seed 233  # set random seed as 233
```

<div align=center>
<video src="https://user-images.githubusercontent.com/28132635/205051392-e3e47ee3-bd18-4cd7-92ac-1cfc66014601.mp4"/>
</div>

If you only want to save video of depth map, you can use the following command:

```shell
python demo/mmagic_inference_demo.py --model-name eg3d \
    --model-config configs/eg3d/eg3d_cvt-official-rgb_shapenet-128x128.py \
    --model-ckpt https://download.openmmlab.com/mmediting/eg3d/eg3d_cvt-official-rgb_shapenet-128x128-85757f4d.pth \
    --result-out-dir eg3d_output \  # save images and videos to `eg3d_output`
    --interpolation camera \  # interpolation camera position only
    --num-images 100 \  # generate 100 images during interpolation
    --vis-mode depth  # only visualize depth image
```

<div align=center>
<video src="https://user-images.githubusercontent.com/28132635/205051103-b0a0e540-c6b8-4f3c-a9ee-0e01ee9fd75b.mp4"/>
</div>

## How to prepare dataset

You should prepare your dataset follow the official [repo](https://github.com/NVlabs/eg3d/tree/main/dataset_preprocessing). Then preprocess the `dataset.json` with the following script:

```python
import json
from argparse import ArgumentParser

from mmengine.fileio.io import load


def main():

    parser = ArgumentParser()
    parser.add_argument(
        'in-anno', type=str, help='Path to the official annotation file.')
    parser.add_argument(
        'out-anno', type=str, help='Path to MMagicing\'s annotation file.')
    args = parser.parse_args()

    anno = load(args.in_anno)
    label = anno['labels']

    anno_dict = {}
    for line in label:
        name, label = line
        anno_dict[name] = label

    with open(args.out_anno, 'w') as file:
        json.dump(anno_dict, file)


if __name__ == '__main__':
    main()
```

## Citation

```latex
@InProceedings{Chan_2022_CVPR,
    author    = {Chan, Eric R. and Lin, Connor Z. and Chan, Matthew A. and Nagano, Koki and Pan, Boxiao and De Mello, Shalini and Gallo, Orazio and Guibas, Leonidas J. and Tremblay, Jonathan and Khamis, Sameh and Karras, Tero and Wetzstein, Gordon},
    title     = {Efficient Geometry-Aware 3D Generative Adversarial Networks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {16123-16133}
}
```
