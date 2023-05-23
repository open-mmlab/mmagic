# Guided Diffusion (NeurIPS'2021)

> [Diffusion Models Beat GANs on Image Synthesis](https://papers.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)

> **Task**: Image Generation

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models. We achieve this on unconditional image synthesis by finding a better architecture through a series of ablations. For conditional image synthesis, we further improve sample quality with classifier guidance: a simple, compute-efficient method for trading off diversity for fidelity using gradients from a classifier. We achieve an FID of 2.97 on ImageNet 128x128, 4.59 on ImageNet 256x256, and 7.72 on ImageNet 512x512, and we match BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution. Finally, we find that classifier guidance combines well with upsampling diffusion models, further improving FID to 3.94 on ImageNet 256x256 and 3.85 on ImageNet 512x512.

<!-- [IMAGE] -->

<div align=center >
 <img src="https://user-images.githubusercontent.com/22982797/204706276-e340c545-3ec6-48bf-be21-58ed44e8a4df.jpg" width="400"/>
</div >

## Results and models

<div align="center">
  <b>hamster, classifier-guidance samplings with CGS=1.0</b>
  <br/>
  <img src="https://user-images.githubusercontent.com/22982797/212831070-470034df-0a9f-4a75-8ab8-97d39bc1806c.png" width="400"/>
</div>

**ImageNet**

|                               Model                                |     Dataset      | Scheduler | Steps | CGS | Time Consuming(A100) | FID-Full-50K |                               Download                                |
| :----------------------------------------------------------------: | :--------------: | :-------: | :---: | :-: | :------------------: | :----------: | :-------------------------------------------------------------------: |
| [adm_ddim250_8xb32_imagenet-64x64](./adm_ddim250_8xb32_imagenet-64x64.py) |  ImageNet 64x64  |   DDIM    |  250  |  -  |          1h          |    3.2284    | [ckpt](https://download.openmmlab.com/mmediting/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth) |
| [adm-g_ddim25_8xb32_imagenet-64x64](configs/guided_diffusion/adm-g_ddim25_8xb32_imagenet-64x64.py) |  ImageNet 64x64  |   DDIM    |  25   | 1.0 |          2h          |    3.7566    | [ckpt](https://download.openmmlab.com/mmediting/guided_diffusion/adm-g_8xb32_imagenet-64x64-2c0fbeda.pth) |
| [adm_ddim250_8xb32_imagenet-256x256](configs/guided_diffusion/adm_ddim250_8xb32_imagenet-256x256.py) | ImageNet 256x256 |   DDIM    |  250  |  -  |          -           |      -       | [ckpt](https://download.openmmlab.com/mmediting/guided_diffusion/adm_8xb32_imagenet-256x256-f94735fe.pth) |
| [adm-g_ddim25_8xb32_imagenet-256x256](configs/guided_diffusion/adm-g_ddim25_8xb32_imagenet-256x256.py) | ImageNet 256x256 |   DDIM    |  25   | 1.0 |          -           |      -       | [ckpt](https://download.openmmlab.com/mmediting/guided_diffusion/adm-g_8xb32_imagenet-256x256-aec3fc9f.pth) |
| [adm_ddim250_8xb32_imagenet-512x512](configs/guided_diffusion/adm_ddim250_8xb32_imagenet-512x512.py) | ImageNet 512x512 |   DDIM    |  250  |  -  |          -           |      -       | [ckpt](https://download.openmmlab.com/mmediting/guided_diffusion/adm-u_8xb32_imagenet-512x512-60b381cb.pth) |
| [adm-g_ddim25_8xb32_imagenet-512x512](configs/guided_diffusion/adm-g_ddim25_8xb32_imagenet-512x512.py) | ImageNet 512x512 |   DDIM    |  25   | 1.0 |          -           |      -       | [ckpt](https://download.openmmlab.com/mmediting/guided_diffusion/adm-g_8xb32_imagenet-512x512-23cf0b58.pth) |

## Quick Start

**infer**

<details>
<summary>Infer Instructions</summary>

You can run adm as follows:

```python
from mmengine import Config, MODELS
from mmengine.registry import init_default_scope
from torchvision.utils import save_image

init_default_scope('mmagic')

# sampling without classifier guidance, CGS=1.0
config = 'configs/guided_diffusion/adm-g_ddim25_8xb32_imagenet-64x64.py'
ckpt_path = 'https://download.openmmlab.com/mmediting/guided_diffusion/adm-g_8xb32_imagenet-64x64-2c0fbeda.pth'  # noqa

model_cfg = Config.fromfile(config).model
model_cfg.pretrained_cfgs = dict(unet=dict(ckpt_path=ckpt_path, prefix='unet'),
                                 classifier=dict(ckpt_path=ckpt_path, prefix='classifier'))
model = MODELS.build(model_cfg).cuda().eval()

samples = model.infer(
            init_image=None,
            batch_size=4,
            num_inference_steps=25,
            labels=333,
            classifier_scale=1.0,
            show_progress=True)['samples']

# sampling without classifier guidance
config = 'configs/guided_diffusion/adm_ddim250_8xb32_imagenet-64x64.py'
ckpt_path = 'https://download.openmmlab.com/mmediting/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth'  # noqa

model_cfg = Config.fromfile(config).model
model_cfg.pretrained_cfgs = dict(unet=dict(ckpt_path=ckpt_path, prefix='unet'))
model = MODELS.build(model_cfg).cuda().eval()

samples = model.infer(
            init_image=None,
            batch_size=4,
            num_inference_steps=250,
            labels=None,
            classifier_scale=0.0,
            show_progress=True)['samples']
```

**Test**

<details>
<summary>Test Instructions</summary>

You can use the following commands to test a model with cpu or single/multiple GPUs.

```shell
# cpu test
CUDA_VISIBLE_DEVICES=-1 python tools/test.py configs/guided_diffusion/adm-u_ddim250_8xb32_imagenet-64x64.py https://download.openmmlab.com/mmgen/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth

# single-gpu test
python tools/test.py configs/guided_diffusion/adm-u_ddim250_8xb32_imagenet-64x64.py https://download.openmmlab.com/mmgen/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth

# multi-gpu test
./tools/dist_test.sh configs/guided_diffusion/adm-u_ddim250_8xb32_imagenet-64x64.py https://download.openmmlab.com/mmgen/guided_diffusion/adm-u-cvt-rgb_8xb32_imagenet-64x64-7ff0080b.pth 8
```

For more details, you can refer to **Test a pre-trained model** part in [train_test.md](/docs/en/user_guides/train_test.md#Test-a-pre-trained-model-in-MMagic).

</details>

## Citation

```bibtex
@article{PrafullaDhariwal2021DiffusionMB,
  title={Diffusion Models Beat GANs on Image Synthesis},
  author={Prafulla Dhariwal and Alex Nichol},
  journal={arXiv: Learning},
  year={2021}
}
```
