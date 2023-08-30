# FastComposer (2023)

> [FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention](https://arxiv.org/abs/2305.10431)

> **Task**: Text2Image

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Diffusion models excel at text-to-image generation, especially in subject-driven generation for personalized images. However, existing methods are inefficient due to the subject-specific fine-tuning, which is computationally intensive and hampers efficient deployment. Moreover, existing methods struggle with multi-subject generation as they often blend features among subjects. We present FastComposer which enables efficient, personalized, multi-subject text-to-image generation without fine-tuning. FastComposer uses subject embeddings extracted by an image encoder to augment the generic text conditioning in diffusion models, enabling personalized image generation based on subject images and textual instructions with only forward passes. To address the identity blending problem in the multi-subject generation, FastComposer proposes cross-attention localization supervision during training, enforcing the attention of reference subjects localized to the correct regions in the target images. Naively conditioning on subject embeddings results in subject overfitting. FastComposer proposes delayed subject conditioning in the denoising step to maintain both identity and editability in subject-driven image generation. FastComposer generates images of multiple unseen individuals with different styles, actions, and contexts. It achieves 300x-2500x speedup compared to fine-tuning-based methods and requires zero extra storage for new subjects. FastComposer paves the way for efficient, personalized, and high-quality multi-subject image creation.

<!-- [IMAGE] -->

<div align=center>
<img src="https://fastcomposer.mit.edu/figures/multi_subject.png">
</div>

## Pretrained models

This model has several weights including vae, unet and clip. You should download the weights from [stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [clipModel](https://huggingface.co/openai/clip-vit-large-patch14),and then change the 'stable_diffusion_v15_url' and 'clip_vit_url' in config to the corresponding weights path and "finetuned_model_path" to the weight path of fastcomposer.

|                    Model                    | Dataset |                                            Download                                             |
| :-----------------------------------------: | :-----: | :---------------------------------------------------------------------------------------------: |
| [FastComposer](./fastcomposer_8xb1_FFHQ.py) |    -    | [model](https://download.openxlab.org.cn/models/xiaomile/fastcomposer/weight/pytorch_model.bin) |

## Quick Start

You can run the demo locally by

```bash
python demo/gradio_fastcomposer.py
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
