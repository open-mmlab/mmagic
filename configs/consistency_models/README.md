# Consistency Models (ICML'2023)

> [Consistency Models](https://arxiv.org/abs/2303.01469)

> **Task**: conditional

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Diffusion models have significantly advanced the fields of image, audio, and video generation, but they depend on an iterative sampling process that causes slow generation. To overcome this limitation, we propose consistency models, a new family of models that generate high quality samples by directly mapping noise to data. They support fast one-step generation by design, while still allowing multistep sampling to trade compute for sample quality. They also support zero-shot data editing, such as image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either by distilling pre-trained diffusion models, or as standalone generative models altogether. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in one- and few-step sampling, achieving the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet 64x64 for one-step generation. When trained in isolation, consistency models become a new family of generative models that can outperform existing one-step, non-adversarial generative models on standard benchmarks such as CIFAR-10, ImageNet 64x64 and LSUN 256x256.

<div align="center">
<img src="https://github.com/xiaomile/mmagic/assets/14927720/1586f0c0-8def-4339-b898-470333a26125" width=800>
</div>

## Pre-trained models

|                                             Model                                             |  Dataset   | Download |
| :-------------------------------------------------------------------------------------------: | :--------: | :------: |
|       [onestep on ImageNet-64](./consistency_models_8xb256-imagenet1k-onestep-64x64.py)       | imagenet1k |    -     |
|     [multistep on ImageNet-64](./consistency_models_8xb256-imagenet1k-multistep-64x64.py)     | imagenet1k |    -     |
|   [onestep on LSUN Bedroom-256](./consistency_models_8xb32-LSUN-bedroom-onestep-256x256.py)   |    LSUN    |    -     |
| [multistep on LSUN Bedroom-256](./consistency_models_8xb32-LSUN-bedroom-multistep-256x256.py) |    LSUN    |    -     |
|       [onstep on LSUN Cat-256](./consistency_models_8xb32-LSUN-cat-onestep-256x256.py)        |    LSUN    |    -     |
|     [multistep on LSUN Cat-256](./consistency_models_8xb32-LSUN-cat-multistep-256x256.py)     |    LSUN    |    -     |

You can also download checkpoints which is the main models in the paper to local machine and deliver the path to 'model_path' before infer.
Here are the download links for each model checkpoint:

- EDM on ImageNet-64: [edm_imagenet64_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_imagenet64_ema.pt)
- CD on ImageNet-64 with l2 metric: [cd_imagenet64_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_l2.pt)
- CD on ImageNet-64 with LPIPS metric: [cd_imagenet64_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_lpips.pt)
- CT on ImageNet-64: [ct_imagenet64.pt](https://openaipublic.blob.core.windows.net/consistency/ct_imagenet64.pt)
- EDM on LSUN Bedroom-256: [edm_bedroom256_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_bedroom256_ema.pt)
- CD on LSUN Bedroom-256 with l2 metric: [cd_bedroom256_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_l2.pt)
- CD on LSUN Bedroom-256 with LPIPS metric: [cd_bedroom256_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt)
- CT on LSUN Bedroom-256: [ct_bedroom256.pt](https://openaipublic.blob.core.windows.net/consistency/ct_bedroom256.pt)
- EDM on LSUN Cat-256: [edm_cat256_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_cat256_ema.pt)
- CD on LSUN Cat-256 with l2 metric: [cd_cat256_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_l2.pt)
- CD on LSUN Cat-256 with LPIPS metric: [cd_cat256_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_lpips.pt)
- CT on LSUN Cat-256: [ct_cat256.pt](https://openaipublic.blob.core.windows.net/consistency/ct_cat256.pt)

## quick start

**Infer**

<details>
<summary>Infer Instructions</summary>

You can use the following commands to infer with the model.

```shell
# onestep
python demo\mmagic_inference_demo.py \
    --model-name consistency_models \
    --model-config configs/consistency_models/consistency_models_8xb256-imagenet1k-onestep-64x64.py \
    --result-out-dir demo_consistency_model.jpg

# multistep
python demo\mmagic_inference_demo.py \
    --model-name consistency_models \
    --model-config configs/consistency_models/consistency_models_8xb256-imagenet1k-multistep-64x64.py \
    --result-out-dir demo_consistency_model.jpg
```

</details>

# Citation

```bibtex
@article{song2023consistency,
  title={Consistency Models},
  author={Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2303.01469},
  year={2023},
}
```
