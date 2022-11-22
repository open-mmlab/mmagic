# Disco Diffusion(Google Colab)

> [](<>)

> **Task**: Text2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Disco Diffusion (DD) is a Google Colab Notebook which leverages an AI Image generating technique called CLIP-Guided Diffusion to allow you to create compelling and beautiful images from just text inputs. Created by Somnai, augmented by Gandamu, and building on the work of RiversHaveWings, nshepperd, and many others.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/22982797/201001789-7ef108a0-f607-401e-98dc-4e16d6be384f.png"/>
</div>

## Models Card

## Quick Start

In order to get started, we introduce a simplest way to get an image within 6 line of codes.

```python
from mmengine import Config, MODELS
from mmedit.utils import register_all_modules
register_all_modules()

disco = MODELS.build(Config.fromfile('configs/disco/disco-baseline.py').model).cuda().eval()
text_prompts = {
    0: ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.", "yellow color scheme"]
}
image = disco.infer(height=768, width=1280, text_prompts=text_prompts, show_progress=True, num_inference_steps=250, eta=0.8)['samples']
```

## Advanced Tutorials

For detailed description and advanced usage.

### Overall Architecture(In Construction)

### Infer Settings

For fixed Disco-Diffusions, there are several runtime settings.

1. Image Resolution.
   Despite the limit of your device limitation, you can set height and width of image as you like.

Performing code,

```python
from mmengine import Config, MODELS
from mmedit.utils import register_all_modules
register_all_modules()

disco = MODELS.build(Config.fromfile('configs/disco/disco-baseline.py').model).cuda().eval()
text_prompts = {
    0: ["A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.", "yellow color scheme"]
}
image = disco.infer(height=512, width=1024, text_prompts=text_prompts, show_progress=True, num_inference_steps=250, eta=0.8)['samples']
```

get

<div align=center>
<img src="https://user-images.githubusercontent.com/22982797/201041058-b47a897c-852e-4b78-9627-48706dade1d5.png"/>
</div>

2. Initial image.
   You can set the initial image for your art work, simply set `init_image` to your image path. By set `init_scale`, you can adjust the similarity of initial image and your result.

**Note**: Make sure you set `skip_steps` to ~50% of your steps if you want to use an init image.

For example, Take this picture as initial image

<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/201272831-81f2b1f4-3e28-4468-8e84-b7c52ad74e11.jpg" width="800"/>
</div>

Note that, `init_scale` need to be set in config, this field is contained in `loss_cfg`.

```python
from mmengine import Config, MODELS
from mmedit.utils import register_all_modules

register_all_modules()
config = 'configs/disco/disco-init_scale20.py'
disco = MODELS.build(Config.fromfile(config).model).cuda().eval()
text_prompts = {
    0: ["a huge dragon, human like, flying with flame, and two big wings"]
}
image_path = 'PATH/TO/INIT_IMAGE'
image = disco.infer(width=1280, height=768, init_image=image_path, text_prompts=text_prompts, show_progress=True, num_inference_steps=250, skip_steps=150, eta=0.8)['samples']
```

and get

<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/201273268-ce775eeb-fb9d-4997-a3f6-b93835593f36.png" width="800"/>
</div>

Then we use default `init_scale=1000`

```python
from mmengine import Config, MODELS
from mmedit.utils import register_all_modules

register_all_modules()
config = 'configs/disco/disco-baseline.py'
disco = MODELS.build(Config.fromfile(config).model).cuda().eval()
text_prompts = {
    0: ["a huge dragon, human like, flying with flame, and two big wings"]
}
image_path = 'PATH/TO/INIT_IMAGE'
image = disco.infer(width=1280, height=768, init_image=image_path, text_prompts=text_prompts, show_progress=True, num_inference_steps=250, skip_steps=150, eta=0.8)['samples']
```

and get

<div align="center">
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/201273252-3e9d1293-5a83-4ca1-a177-b9fa2639ba14.png" width="800"/>
</div>

### Unet Settings(In Construction)

### Clip Models Settings(In Construction)

### Cutter Settings(In Construction)

### Diffuser Settings(In Construction)

### Loss Settings(In Construction)

## Citation

```bibtex
@misc{github,
  author={alembics},
  title={disco-diffusion},
  year={2022},
  url={https://github.com/alembics/disco-diffusion},
}
```
