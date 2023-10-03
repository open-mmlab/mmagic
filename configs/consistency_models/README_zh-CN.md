# Consistency Models (ICML'2023)

> [Consistency Models](https://arxiv.org/abs/2303.01469)

> **任务**: 条件生成

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

扩散模型在图像、音频和视频生成领域取得了显著的进展，但它们依赖于迭代采样过程，导致生成速度较慢。为了克服这个限制，我们提出了一种新的模型家族——一致性模型，通过直接将噪声映射到数据来生成高质量的样本。它们通过设计支持快速的单步生成，同时仍然允许多步采样以在计算和样本质量之间进行权衡。它们还支持零样本数据编辑，如图像修补、上色和超分辨率，而不需要在这些任务上进行显式训练。一致性模型可以通过蒸馏预训练的扩散模型或作为独立的生成模型进行训练。通过大量实验证明，它们在单步和少步采样方面优于现有的扩散模型蒸馏技术，实现了 CIFAR-10 上的新的最先进 FID（Fréchet Inception Distance）为 3.55，ImageNet 64x64 上为 6.20 的结果。当独立训练时，一致性模型成为一种新的生成模型家族，在 CIFAR-10、ImageNet 64x64 和 LSUN 256x256 等标准基准测试上可以优于现有的单步非对抗性生成模型。

## 预训练模型

我们已经发布了论文中主要模型的权重。
以下是每个模型权重的下载链接：

|                                           Model                                            |  Dataset   | metric |                                            Download                                            |
| :----------------------------------------------------------------------------------------: | :--------: | :----: | :--------------------------------------------------------------------------------------------: |
|       [EDM on ImageNet-64](./consistency_models_8xb256-imagenet1k-onestep-64x64.py)        | imagenet1k |   -    | [edm_imagenet64_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_imagenet64_ema.pt) |
| [CD on ImageNet-64 with l2 metric](./consistency_models_8xb256-imagenet1k-onestep-64x64.py) | imagenet1k |   l2   | [cd_imagenet64_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_l2.pt) |
| [CD on ImageNet-64 with LPIPS metric](./consistency_models_8xb256-imagenet1k-onestep-64x64.py) | imagenet1k | lpips  | [cd_imagenet64_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_imagenet64_lpips.pt) |
|        [CT on ImageNet-64](./consistency_models_8xb256-imagenet1k-onestep-64x64.py)        | imagenet1k |   -    |  [ct_imagenet64.pt](https://openaipublic.blob.core.windows.net/consistency/ct_imagenet64.pt)   |
|   [EDM on LSUN Bedroom-256](./consistency_models_8xb32-LSUN-bedroom-onestep-256x256.py)    |    LSUN    |   -    | [edm_bedroom256_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_bedroom256_ema.pt) |
| [CD on LSUN Bedroom-256 with l2 metric](./consistency_models_8xb32-LSUN-bedroom-onestep-256x256.py) |    LSUN    |   l2   | [cd_bedroom256_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_l2.pt) |
| [CD on LSUN Bedroom-256 with LPIPS metric](./consistency_models_8xb32-LSUN-bedroom-onestep-256x256.py) |    LSUN    | lpips  | [cd_bedroom256_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt) |
|    [CT on LSUN Bedroom-256](./consistency_models_8xb32-LSUN-bedroom-onestep-256x256.py)    |    LSUN    |   -    |  [ct_bedroom256.pt](https://openaipublic.blob.core.windows.net/consistency/ct_bedroom256.pt)   |
|       [EDM on LSUN Cat-256](./consistency_models_8xb32-LSUN-cat-onestep-256x256.py)        |    LSUN    |   -    | [edm_cat256_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_cat256_ema.pt)  |
| [CD on LSUN Cat-256 with l2 metric](./consistency_models_8xb32-LSUN-cat-onestep-256x256.py) |    LSUN    |   l2   |   [cd_cat256_l2.pt](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_l2.pt)    |
| [CD on LSUN Cat-256 with LPIPS metric](./consistency_models_8xb32-LSUN-cat-onestep-256x256.py) |    LSUN    | lpips  | [cd_cat256_lpips.pt](https://openaipublic.blob.core.windows.net/consistency/cd_cat256_lpips.pt) |
|        [CT on LSUN Cat-256](./consistency_models_8xb32-LSUN-cat-onestep-256x256.py)        |    LSUN    |   -    |      [ct_cat256.pt](https://openaipublic.blob.core.windows.net/consistency/ct_cat256.pt)       |

## 快速开始

**推理**

<details>
<summary>推理说明</summary>

您可以使用以下命令来使用该模型进行推理：

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
