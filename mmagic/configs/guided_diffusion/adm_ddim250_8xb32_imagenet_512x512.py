# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.imagenet_512 import *
    from .._base_.gen_default_runtime import *

from mmagic.evaluation.metrics import FrechetInceptionDistance
from mmagic.models.data_preprocessors.data_preprocessor import DataPreprocessor
from mmagic.models.diffusion_schedulers.ddim_scheduler import EditDDIMScheduler
from mmagic.models.editors.ddpm.denoising_unet import (DenoisingUnet,
                                                       MultiHeadAttentionBlock)
from mmagic.models.editors.guided_diffusion import AblatedDiffusionModel

model = dict(
    type=AblatedDiffusionModel,
    data_preprocessor=dict(type=DataPreprocessor),
    unet=dict(
        type=DenoisingUnet,
        image_size=512,
        in_channels=3,
        base_channels=256,
        resblocks_per_downsample=2,
        attention_res=(32, 16, 8),
        norm_cfg=dict(type='GN32', num_groups=32),
        dropout=0.1,
        num_classes=1000,
        use_fp16=False,
        resblock_updown=True,
        attention_cfg=dict(
            type=MultiHeadAttentionBlock,
            num_heads=4,
            num_head_channels=64,
            use_new_attention_order=False),
        use_scale_shift_norm=True),
    diffusion_scheduler=dict(
        type=EditDDIMScheduler,
        variance_type='learned_range',
        beta_schedule='linear'),
    rgb2bgr=True,
    use_fp16=False)

test_dataloader.update(dict(batch_size=32, num_workers=8))
train_cfg = dict(max_iters=100000)
metrics = [
    dict(
        type=FrechetInceptionDistance,
        prefix='FID-Full-50k',
        fake_nums=50000,
        inception_style='StyleGAN',
        sample_model='orig',
        sample_kwargs=dict(
            num_inference_steps=250, show_progress=True, classifier_scale=1.))
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)
