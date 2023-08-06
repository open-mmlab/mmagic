# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.ddpm.denoising_unet import (DenoisingUnet,
                                                       NormWithEmbedding)


def test_DenoisingUnet_Target():
    input = torch.rand((2, 4, 64, 64))
    encoder = torch.rand((2, 77, 768))
    unet = DenoisingUnet(
        image_size=128,
        base_channels=32,
        channels_cfg=[1, 2],
        unet_type='stable',
        act_cfg=dict(type='silu', inplace=False),
        cross_attention_dim=768,
        num_heads=2,
        in_channels=4,
        layers_per_block=1,
        down_block_types=['CrossAttnDownBlock2D', 'DownBlock2D'],
        up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'],
        output_cfg=dict(var='fixed'))
    output = unet.forward(input, 1, encoder)
    assert output['sample'].shape == (2, 4, 64, 64)

    input = torch.rand((2, 4, 64, 64))
    encoder = torch.rand((2, 77, 1024))
    unet = DenoisingUnet(
        image_size=32,
        base_channels=32,
        channels_cfg=[1, 2],
        unet_type='stable',
        act_cfg=dict(type='silu', inplace=False),
        cross_attention_dim=1024,
        num_heads=2,
        in_channels=4,
        layers_per_block=1,
        down_block_types=['CrossAttnDownBlock2D', 'DownBlock2D'],
        up_block_types=['UpBlock2D', 'CrossAttnUpBlock2D'],
        output_cfg=dict(var='fixed'))
    output = unet.forward(input, 1, encoder)
    assert output['sample'].shape == (2, 4, 64, 64)


def test_NormWithEmbedding():
    input = torch.rand((4, 32))
    emb = torch.rand((4, 32))
    ins = NormWithEmbedding(32, 32)
    output = ins.forward(input, emb)
    assert output.shape == (4, 32, 4, 32)
