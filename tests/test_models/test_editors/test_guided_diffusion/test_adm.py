# Copyright (c) OpenMMLab. All rights reserved.
from mmedit.models.editors import AblatedDiffusionModel
import torch

def test_ablated_diffusion_model(self):
    data_preprocessor = dict(
        type='EditDataPreprocessor', mean=[127.5], std=[127.5])
    unet = dict(
        type='DenoisingUnet',
        image_size=64,
        in_channels=3,
        base_channels=256,
        resblocks_per_downsample=2,
        attention_res=(32, 16, 8),
        norm_cfg=dict(type='GN32', num_groups=32),
        dropout=0.0,
        num_classes=1000,
        use_fp16=False,
        resblock_updown=True,
        attention_cfg=dict(
            type='MultiHeadAttentionBlock',
            num_heads=4,
            num_head_channels=64,
            use_new_attention_order=False),
        use_scale_shift_norm=True)

    diffuser = dict(
        type='DDPMDiffuser', variance_type='learned_range', beta_schedule='linear')

    adm = AblatedDiffusionModel(
        data_preprocessor,
        unet,
        diffuser,
        use_fp16=False).float().eval()

    if torch.cuda.is_available:
        adm = adm.cuda()

    with torch.no_grad():
        samples = adm.infer(batch_size=2, show_progress=False)['samples']
    
    assert samples.shape == (2, 3, 64, 64)

    
