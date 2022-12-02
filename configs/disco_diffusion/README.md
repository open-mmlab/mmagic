# Disco Diffusion

> [Disco Diffusion](https://github.com/alembics/disco-diffusion)

> **Task**: Text2Image, Image2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Disco Diffusion (DD) is a Google Colab Notebook which leverages an AI Image generating technique called CLIP-Guided Diffusion to allow you to create compelling and beautiful images from text inputs.

Created by Somnai, augmented by Gandamu, and building on the work of RiversHaveWings, nshepperd, and many others.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/22982797/204526957-ac30547e-5a44-417a-aaa2-6b357b4a139c.png" width="400"/>
</div >

## Results and models

We have converted several `unet` weights and offer related configs. Or usage of different `unet`, please refer to tutorial.

| Diffusion Model                          | Config                                                                      | Weights                                                                               |
| ---------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 512x512_diffusion_uncond_finetune_008100 | [config](configs/disco/disco-diffusion_adm-u-finetuned_imagenet-512x512.py) | [weights](https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u_finetuned_imagenet-512x512-ab471d70.pth) |
| 256x256_diffusion_uncond                 | [config](configs/disco/disco-diffusion_adm-u-finetuned_imagenet-256x256.py) | [weights](<>)                                                                         |
| portrait_generator_v001                  | [config](configs/disco/disco-diffusion_portrait_generator_v001.py)          | [weights](https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u-cvt-rgb_portrait-v001-f4a3f3bc.pth) |
| pixelartdiffusion_expanded               | Coming soon!                                                                |                                                                                       |
| pixel_art_diffusion_hard_256             | Coming soon!                                                                |                                                                                       |
| pixel_art_diffusion_soft_256             | Coming soon!                                                                |                                                                                       |
| pixelartdiffusion4k                      | Coming soon!                                                                |                                                                                       |
| watercolordiffusion_2                    | Coming soon!                                                                |                                                                                       |
| watercolordiffusion                      | Coming soon!                                                                |                                                                                       |
| PulpSciFiDiffusion                       | Coming soon!                                                                |                                                                                       |

## To-do List

- [ ] pixelart, watercolor, sci-fiction diffusion models
- [ ] image prompt
- [ ] video generation
- [ ] faster sampler(plms, dpm-solver etc.)

We really welcome community users supporting these items and any other interesting staffs!

## Quick Start

Running the following codes, you can get a text-generated image.

```python
from mmengine import Config, MODELS
from mmedit.utils import register_all_modules
from torchvision.utils import save_image

register_all_modules()

disco = MODELS.build(
    Config.fromfile('configs/disco/disco-baseline.py').model).cuda().eval()
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

## Tutorials

Coming soon!

## Credits

Since our adaptation of disco-diffusion are heavily influenced by disco [colab](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb#scrollTo=License), here we copy the credits below.

<details>
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
