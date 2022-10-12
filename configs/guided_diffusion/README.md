# Guided Diffusion (NeurIPS'2021)

> [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf)

> **Task**: Diffusion Models

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->
We show that diffusion models can achieve image sample quality superior to thecurrent state-of-the-art generative models. We achieve this on unconditional im-age synthesis by finding a better architecture through a series of ablations.  Forconditional image synthesis, we further improve sample quality with classifier guid-ance: a simple, compute-efficient method for trading off diversity for fidelity usinggradients from a classifier.  We achieve an FID of 2.97 on ImageNet 128×128,4.59 on ImageNet 256×256,  and 7.72 on ImageNet 512×512,  and we matchBigGAN-deep even with as few as 25 forward passes per sample, all while main-taining better coverage of the distribution. Finally, we find that classifier guidancecombines well with upsampling diffusion models, further improving FID to 3.94on ImageNet 256×256 and 3.85 on ImageNet 512×512. We release our code at https://github.com/openai/guided-diffusion.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/22982797/195326738-dc132051-fbb8-4dc1-b9a7-65c1d3ec57f4.png" width="400"/>
</div >

## Results and models

**ImageNet-1k**


## Quick Start

**Infer**

You can use the following codes to sample an image.
```python
from mmedit.registry import MODELS
config = '
adm = MODELS.build(config).cuda().float().eval()
adm.load_state_dict(torch.load(''))
batch_size = 2
with torch.no_grad():
    images = adm.infer(batch_size=batch_size, show_progress=True)['samples']
```

**Train**

<details>
<summary>Train Instructions</summary>

You can use the following commands to train a model with cpu or single/multiple GPUs.

```shell
# cpu train
CUDA_VISIBLE_DEVICES=-1 python tools/train.py configs/guided_diffusion/adm-u_8xb32_imagenet-64x64.py

# single-gpu train
python tools/train.py configs/guided_diffusion/adm-u_8xb32_imagenet-64x64.py

# multi-gpu train
./tools/dist_train.sh configs/guided_diffusion/adm-u_8xb32_imagenet-64x64.py 8
```

For more details, you can refer to **Train a model** part in [train_test.md](/docs/en/user_guides/train_test.md#Train-a-model-in-MMEditing).

</details>

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/guided_diffusion/adm-u_8xb32_imagenet-64x64.py

# single-gpu test
python tools/test.py configs/guided_diffusion/adm-u_8xb32_imagenet-512x512.py

# multi-gpu test
./tools/dist_test.sh configs/guided_diffusion/adm-u_ddim250_8xb32_imagenet-512x512.py
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMEditing).

</details>


## Citation

```bibtex
@article{dhariwal2021diffusion,
  title={Diffusion models beat gans on image synthesis},
  author={Dhariwal, Prafulla and Nichol, Alexander},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={8780--8794},
  year={2021}
}
```
