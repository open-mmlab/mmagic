# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.models.utils import build_module
from mmagic.registry import MODELS

stable_diffusion_v15_url = 'runwayml/stable-diffusion-v1-5'
unet_cfg = dict(
    type='UNet3DConditionMotionModel',
    unet_use_cross_frame_attention=False,
    unet_use_temporal_attention=False,
    use_motion_module=True,
    motion_module_resolutions=[1, 2, 4, 8],
    motion_module_mid_block=False,
    motion_module_decoder_only=False,
    motion_module_type='Vanilla',
    motion_module_kwargs=dict(
        num_attention_heads=8,
        num_transformer_block=1,
        attention_block_types=['Temporal_Self', 'Temporal_Self'],
        temporal_position_encoding=True,
        temporal_position_encoding_max_len=24,
        temporal_attention_dim_div=1),
    subfolder='unet')


@pytest.mark.skipif(
    'win' in platform.system().lower()
    or digit_version(TORCH_VERSION) <= digit_version('1.9.2'),
    reason='skip on windows due to limited RAM'
    'and torch >= 1.10.0')
def test_Unet3D():
    input = torch.rand((1, 4, 16, 8, 8))
    text_feat = torch.rand([1, 20, 768])
    unet = build_module(unet_cfg, MODELS)
    output = unet.forward(input, 10, text_feat)
    assert output['sample'].shape == (1, 4, 16, 8, 8)


if __name__ == '__main__':
    test_Unet3D()
