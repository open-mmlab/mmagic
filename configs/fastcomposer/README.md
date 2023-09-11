# FastComposer (2023)

> [FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention](https://arxiv.org/abs/2305.10431)

> **Task**: Text2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Diffusion models excel at text-to-image generation, especially in subject-driven generation for personalized images. However, existing methods are inefficient due to the subject-specific fine-tuning, which is computationally intensive and hampers efficient deployment. Moreover, existing methods struggle with multi-subject generation as they often blend features among subjects. We present FastComposer which enables efficient, personalized, multi-subject text-to-image generation without fine-tuning. FastComposer uses subject embeddings extracted by an image encoder to augment the generic text conditioning in diffusion models, enabling personalized image generation based on subject images and textual instructions with only forward passes. To address the identity blending problem in the multi-subject generation, FastComposer proposes cross-attention localization supervision during training, enforcing the attention of reference subjects localized to the correct regions in the target images. Naively conditioning on subject embeddings results in subject overfitting. FastComposer proposes delayed subject conditioning in the denoising step to maintain both identity and editability in subject-driven image generation. FastComposer generates images of multiple unseen individuals with different styles, actions, and contexts. It achieves 300x-2500x speedup compared to fine-tuning-based methods and requires zero extra storage for new subjects. FastComposer paves the way for efficient, personalized, and high-quality multi-subject image creation.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/14927720/265914135-8a25789c-8d30-40cb-8ac5-e3bd3b617aac.png">
</div>

## Pretrained models

This model has several weights including vae, unet and clip. You should download the weights from [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [clipModel](https://huggingface.co/openai/clip-vit-large-patch14),and then change the 'stable_diffusion_v15_url' and 'clip_vit_url' in config to the corresponding weights path and "finetuned_model_path" to the weight path of fastcomposer.

|                    Model                     | Dataset |                                            Download                                             |
| :------------------------------------------: | :-----: | :---------------------------------------------------------------------------------------------: |
| [FastComposer](./fastcomposer_8xb16_FFHQ.py) |    -    | [model](https://download.openxlab.org.cn/models/xiaomile/fastcomposer/weight/pytorch_model.bin) |

## Quick Start

You can run the demo locally by

```bash
python demo/gradio_fastcomposer.py
```

Or running the following codes, you can get a text-generated image.

```python
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules
import torch, gc

gc.collect()
torch.cuda.empty_cache()

register_all_modules()

cfg_file = Config.fromfile('configs/fastcomposer/fastcomposer_8xb16_FFHQ.py')

fastcomposer = MODELS.build(cfg_file.model).cuda()

prompt = "A man img and a man img sitting in a park"
negative_prompt = "((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
alpha_ = 0.75
guidance_scale = 5
num_steps = 50
num_images = 1
image = []
seed = -1

image1 = mmcv.imread('https://user-images.githubusercontent.com/14927720/265911400-91635451-54b6-4dc6-92a7-c1d02f88b62e.jpeg')
image2 = mmcv.imread('https://user-images.githubusercontent.com/14927720/265911502-66b67f53-dff0-4d25-a9af-3330e446aa48.jpeg')

image.append(Image.fromarray(image1))
image.append(Image.fromarray(image2))

if len(image) == 0:
    raise Exception("You need to upload at least one image.")

num_subject_in_text = (
        np.array(fastcomposer.special_tokenizer.encode(prompt))
        == fastcomposer.image_token_id
).sum()
if num_subject_in_text != len(image):
    raise Exception(f"Number of subjects in the text description doesn't match the number of reference images, #text subjects: {num_subject_in_text} #reference image: {len(image)}",
    )

if seed == -1:
    seed = np.random.randint(0, 1000000)

device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'cpu')
generator = torch.Generator(device=device)
generator.manual_seed(seed)

output_dict = fastcomposer.infer(prompt,
                                 negative_prompt=negative_prompt,
                                 height=512,
                                 width=512,
                                 num_inference_steps=num_steps,
                                 guidance_scale=guidance_scale,
                                 num_images_per_prompt=num_images,
                                 generator=generator,
                                 alpha_=alpha_,
                                 reference_subject_images=image)

samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'sample_{idx}.png')
```

## Citation

```bibtex
@article{xiao2023fastcomposer,
            title={FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention},
            author={Xiao, Guangxuan and Yin, Tianwei and Freeman, William T. and Durand, Frédo and Han, Song},
            journal={arXiv},
            year={2023}
          }
```
