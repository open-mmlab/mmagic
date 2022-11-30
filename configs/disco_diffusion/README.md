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

| Diffusion Model                          | Config | Weights |
|------------------------------------------|--------|---------|
| 512x512_diffusion_uncond_finetune_008100 |[config](configs/disco/disco-diffusion_adm-u-finetuned_imagenet-512x512.py)|[weights](https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u_finetuned_imagenet-512x512-ab471d70.pth)|
| 256x256_diffusion_uncond                 |[config](configs/disco/disco-diffusion_adm-u-finetuned_imagenet-256x256.py)|[weights]()|
| portrait_generator_v001                  |[config](configs/disco/disco-diffusion_portrait_generator_v001.py)|[weights](https://download.openmmlab.com/mmediting/synthesizers/disco/adm-u-cvt-rgb_portrait-v001-f4a3f3bc.pth)|
| pixelartdiffusion_expanded               |        Coming soon!     |    
| pixel_art_diffusion_hard_256             |        Coming soon!     |   
| pixel_art_diffusion_soft_256             |        Coming soon!     |     
| pixelartdiffusion4k                      |        Coming soon!     |     
| watercolordiffusion_2                    |        Coming soon!     |     
| watercolordiffusion                      |        Coming soon!     |     
| PulpSciFiDiffusion                       |        Coming soon!     |     

## TO-dO List

-[ ] pixelart, watercolor, sci-fiction diffusion models
-[ ] image prompt
-[ ] video generation
-[ ] fast sampler(plms, dpm-solver etc.)

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
Coming soon!

## Citation
```bibtex
@misc{github,
  author={alembics},
  title={disco-diffusion},
  year={2022},
  url={https://github.com/alembics/disco-diffusion},
}
```
