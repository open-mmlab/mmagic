# SinGAN (ICCV'2019)

> [Singan: Learning a Generative Model from a Single Natural Image](https://openaccess.thecvf.com/content_ICCV_2019/html/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.html)

> **Task**: Internal Learning

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We introduce SinGAN, an unconditional generative model that can be learned from a single natural image. Our model is trained to capture the internal distribution of patches within the image, and is then able to generate high quality, diverse samples that carry the same visual content as the image. SinGAN contains a pyramid of fully convolutional GANs, each responsible for learning the patch distribution at a different scale of the image. This allows generating new samples of arbitrary size and aspect ratio, that have significant variability, yet maintain both the global structure and the fine textures of the training image. In contrast to previous single image GAN schemes, our approach is not limited to texture images, and is not conditional (i.e. it generates samples from noise). User studies confirm that the generated samples are commonly confused to be real images. We illustrate the utility of SinGAN in a wide range of image manipulation tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143054395-000ceec1-3be9-4447-b4b9-effc9de94c62.JPG"/>
</div>

## Results and Models

<div align="center">
  <b> SinGAN balloons</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/12726765/113702715-7861a900-970c-11eb-9dd8-0743cc30701f.png" width="800"/>
</div>

|             Model              |                                     Dataset                                     | Num Scales |                                     Download                                      |
| :----------------------------: | :-----------------------------------------------------------------------------: | :--------: | :-------------------------------------------------------------------------------: |
| [SinGAN](./singan_balloons.py) | [balloons.png](https://download.openmmlab.com/mmediting/dataset/singan/balloons.png) |     8      | [ckpt](https://download.openmmlab.com/mmediting/singan/singan_balloons_20210406_191047-8fcd94cf.pth) \| [pkl](https://download.openmmlab.com/mmediting/singan/singan_balloons_20210406_191047-8fcd94cf.pkl) |
|   [SinGAN](./singan_fish.py)   | [fish.jpg](https://download.openmmlab.com/mmediting/dataset/singan/fish-crop.jpg) |     10     | [ckpt](https://download.openmmlab.com/mmediting/singan/singan_fis_20210406_201006-860d91b6.pth) \| [pkl](https://download.openmmlab.com/mmediting/singan/singan_fis_20210406_201006-860d91b6.pkl) |
| [SinGAN](./singan_bohemian.py) | [bohemian.png](https://download.openmmlab.com/mmediting/dataset/singan/bohemian.png) |     10     | [ckpt](https://download.openmmlab.com/mmediting/singan/singan_bohemian_20210406_175439-f964ee38.pth) \| [pkl](https://download.openmmlab.com/mmediting/singan/singan_bohemian_20210406_175439-f964ee38.pkl) |

## Notes for using SinGAN

When training SinGAN models, users may obtain the number of scales (stages) in advance via the following commands. This number is important for constructing config file, which is related to the generator, discriminator, the training iterations and so on.

```shell
>>> from mmgen.datasets.singan_dataset import create_real_pyramid
>>> import mmcv
>>> real = mmcv.imread('real_img_path')
>>> _, _, num_scales = create_real_pyramid(real, min_size=25, max_size=300, scale_factor_init=0.75)
```

When testing SinGAN models, users have to modify the config file to add the `test_cfg`. As shown in `configs/singan/singan_balloons.py`, the only thing you need to do is add the path for `pkl` data. There are some important data containing in the pickle files which you can download from our website.

```python
test_cfg = dict(
    _delete_ = True
    pkl_data = 'path to pkl data'
)
```

## Citation

```latex
@inproceedings{shaham2019singan,
  title={Singan: Learning a generative model from a single natural image},
  author={Shaham, Tamar Rott and Dekel, Tali and Michaeli, Tomer},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4570--4580},
  year={2019},
  url={https://openaccess.thecvf.com/content_ICCV_2019/html/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.html},
}
```
