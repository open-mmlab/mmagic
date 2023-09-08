# FastComposer (2023)

> [FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention](https://arxiv.org/abs/2305.10431)

> **任务**: 文本转图像

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

扩散模型在文本到图像生成方面表现出色，尤其在以主题驱动的个性化图像生成方面。然而，现有方法由于主题特定的微调而效率低下，因为需要大量的计算资源，而这限制了扩散模型高效部署的可能性。此外，现有方法在多主题生成方面存在困难，因为它们经常在不同主题之间混合特征。因此我们提出了FastComposer，它可以实现高效、个性化、多主题的文本到图像生成，而无需进行微调。FastComposer利用图像编码器提取的主题嵌入来增强扩散模型中的通用文本条件，只需进行前向传递即可基于主题图像和文本指令进行个性化图像生成。为了解决多主题生成中的身份混合问题，FastComposer在训练过程中提出了交叉注意力定位监督，强制参考主题的注意力定位于目标图像中的正确区域。简单地基于主题嵌入进行条件设定会导致主题过度拟合的问题。为了在以主题驱动的图像生成中同时保持身份和可编辑性，FastComposer在去噪步骤中提出了延迟主题条件设定的方法。FastComposer可以生成具有不同风格、动作和背景的多个未知个体的图像。与基于微调的方法相比，它实现了300倍到2500倍的加速，并且对于新主题不需要额外的存储空间。正因如此FastComposer为高效、个性化和高质量的多主题图像创作铺平了道路。

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/14927720/265914135-8a25789c-8d30-40cb-8ac5-e3bd3b617aac.png">
</div>

## 预训练模型

该模型有几个权重，包括vae，unet和clip。您应该先从[stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 和 [clipModel](https://huggingface.co/openai/clip-vit-large-patch14) 下载权重，然后将配置中的“stable_diffusion_v15_url”和”clip_vit_url“更改为对应的权重路径，将”finetuned_model_path“更改为fastcomposer的权重路径。

|                    Model                     | Dataset |                                            Download                                             |
| :------------------------------------------: | :-----: | :---------------------------------------------------------------------------------------------: |
| [FastComposer](./fastcomposer_8xb16_FFHQ.py) |    -    | [model](https://download.openxlab.org.cn/models/xiaomile/fastcomposer/weight/pytorch_model.bin) |

## 快速开始

您可以通过以下方式在本地运行来演示

```bash
python demo/gradio_fastcomposer.py
```

或者运行一下代码，您就能获得依照文本生成的特定图像。

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

## 引用

```bibtex
@article{xiao2023fastcomposer,
            title={FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention},
            author={Xiao, Guangxuan and Yin, Tianwei and Freeman, William T. and Durand, Frédo and Han, Song},
            journal={arXiv},
            year={2023}
          }
```
