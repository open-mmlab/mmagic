# Disco Diffusion (2022)

> [Disco Diffusion](https://github.com/alembics/disco-diffusion)

> **任务**: 图文生成, 图像到图像的翻译, 扩散模型

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

Disco Diffusion(DD)是一个 Google Colab 笔记本，它利用一种叫做 CLIP-Guided Diffusion 的人工智能图像生成技术，让你从文本输入中创造出引人注目的精美图像。

由 Somnai 创建，由 Gandamu 改进，并建立在 RiversHaveWings、nshepperd 和许多其他人的工作之上。更多细节见[Credits](#credits)。

<!-- [IMAGE] -->

<table align="center">
<thead>
  <tr>
    <td>
<div align="center">

<img src="https://user-images.githubusercontent.com/22982797/204526957-ac30547e-5a44-417a-aaa2-6b357b4a139c.png" width="400"/>
</div></td>
    <td>
<div align="center">

<img src="https://user-images.githubusercontent.com/22982797/215749979-1ea973c4-3e76-4204-9fa0-b0adf3e942b6.png" width="400"/>
</div></td>
    <td>
<div align="center">

<img src="https://user-images.githubusercontent.com/22982797/215757871-d38e1b78-fee0-4351-be61-5b1e782d1e6e.png" width="400"/>
</div></td>
  </tr>
</thead>
</table>

## 模型与结果

我们已经转换了几个 `unet` 的权重，并提供相关的配置文件。在[Tutorial](#tutorials)中可以看到更多关于不同 `unet` 的细节。

|                                               模型                                               |  数据集  |                                               下载                                               |
| :----------------------------------------------------------------------------------------------: | :------: | :----------------------------------------------------------------------------------------------: |
| [512x512_diffusion_uncond_finetune_008100](./disco-diffusion_adm-u-finetuned_imagenet-512x512.py) | ImageNet | [model](https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u_finetuned_imagenet-512x512-ab471d70.pth) |
|        [256x256_diffusion_uncond](./disco-diffusion_adm-u-finetuned_imagenet-256x256.py)         | ImageNet |                                           [model](<>)                                            |
|             [portrait_generator_v001](./disco-diffusion_portrait-generator-v001.py)              | unknown  | [model](https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u-cvt-rgb_portrait-v001-f4a3f3bc.pth) |

<!-- SKIP THIS TABLE -->

|             模型             |     下载     |
| :--------------------------: | :----------: |
|  pixelartdiffusion_expanded  | Coming soon! |
| pixel_art_diffusion_hard_256 | Coming soon! |
| pixel_art_diffusion_soft_256 | Coming soon! |
|     pixelartdiffusion4k      | Coming soon! |
|    watercolordiffusion_2     | Coming soon! |
|     watercolordiffusion      | Coming soon! |
|      PulpSciFiDiffusion      | Coming soon! |

## 待办列表

- [x] 图文生成
- [x] 图像到图像的翻译
- [x] Imagenet, portrait 扩散模型
- \[\] 像素艺术，水彩，科幻小说的扩散模型
- \[\] 支持图像提示
- \[\] 支持视频生成
- \[\] 支持更快的采样器(plms，dpm-solver等)

我们很欢迎社区用户支持这些项目和任何其他有趣的工作!

## Quick Start

运行以下代码，你可以使用文本生成图像。

```python
from mmengine import Config, MODELS
from mmagic.utils import register_all_modules
from torchvision.utils import save_image

register_all_modules()

disco = MODELS.build(
    Config.fromfile('configs/disco_diffusion/disco-baseline.py').model).cuda().eval()
text_prompts = {
    0: [
        "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.",
        "yellow color scheme"
    ]
}
image = disco.infer(
    height=768,
    width=1280,
    text_prompts=text_prompts,
    show_progress=True,
    num_inference_steps=250,
    eta=0.8)['samples']
save_image(image, "image.png")

```

## 教程

考虑到`disco-diffusion`包含许多可调整的参数，我们为用户提供了一个[jupyter-notebook](./tutorials.ipynb)/[colab](https://githubtocolab.com/open-mmlab/mmagic/blob/main/configs/disco_diffusion/tutorials.ipynb)的教程，展示了不同参数的含义，并给出相应的调整结果。
请参考[Disco Sheet](https://docs.google.com/document/d/1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g/edit)。

## 鸣谢

Since our adaptation of disco-diffusion are heavily influenced by disco [colab](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb#scrollTo=License), here we copy the credits below.

<details>
<summary>鸣谢</summary>
Original notebook by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings). It uses either OpenAI's 256x256 unconditional ImageNet or Katherine Crowson's fine-tuned 512x512 diffusion model (https://github.com/openai/guided-diffusion), together with CLIP (https://github.com/openai/CLIP) to connect text prompts with images.

Modified by Daniel Russell (https://github.com/russelldc, https://twitter.com/danielrussruss) to include (hopefully) optimal params for quick generations in 15-100 timesteps rather than 1000, as well as more robust augmentations.

Further improvements from Dango233 and nshepperd helped improve the quality of diffusion in general, and especially so for shorter runs like this notebook aims to achieve.

Vark added code to load in multiple Clip models at once, which all prompts are evaluated against, which may greatly improve accuracy.

The latest zoom, pan, rotation, and keyframes features were taken from Chigozie Nri's VQGAN Zoom Notebook (https://github.com/chigozienri, https://twitter.com/chigozienri)

Advanced DangoCutn Cutout method is also from Dango223.

\--

Disco:

Somnai (https://twitter.com/Somnai_dreams) added Diffusion Animation techniques, QoL improvements and various implementations of tech and techniques, mostly listed in the changelog below.

3D animation implementation added by Adam Letts (https://twitter.com/gandamu_ml) in collaboration with Somnai. Creation of disco.py and ongoing maintenance.

Turbo feature by Chris Allen (https://twitter.com/zippy731)

Improvements to ability to run on local systems, Windows support, and dependency installation by HostsServer (https://twitter.com/HostsServer)

VR Mode by Tom Mason (https://twitter.com/nin_artificial)

Horizontal and Vertical symmetry functionality by nshepperd. Symmetry transformation_steps by huemin (https://twitter.com/huemin_art). Symmetry integration into Disco Diffusion by Dmitrii Tochilkin (https://twitter.com/cut_pow).

Warp and custom model support by Alex Spirin (https://twitter.com/devdef).

Pixel Art Diffusion, Watercolor Diffusion, and Pulp SciFi Diffusion models from KaliYuga (https://twitter.com/KaliYuga_ai). Follow KaliYuga's Twitter for the latest models and for notebooks with specialized settings.

Integration of OpenCLIP models and initiation of integration of KaliYuga models by Palmweaver / Chris Scalf (https://twitter.com/ChrisScalf11)

Integrated portrait_generator_v001 from Felipe3DArtist (https://twitter.com/Felipe3DArtist)

</details>

## Citation

```bibtex
@misc{github,
  author={alembics},
  title={disco-diffusion},
  year={2022},
  url={https://github.com/alembics/disco-diffusion},
}
```
