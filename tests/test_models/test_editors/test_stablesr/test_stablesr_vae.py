# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.stable_diffusion.vae import AutoencoderKL
from mmagic.models.editors.stablesr.vqgan import AutoencoderKL_Resi


def test_vqmodel():
    input = torch.rand((2, 3, 512, 512))
    vae = AutoencoderKL_Resi(
        latent_channels=4,
        down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D',
                          'DownEncoderBlock2D', 'DownEncoderBlock2D'),
        up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D',
                        'UpDecoderBlock2D', 'UpDecoderBlock2D'),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
    )
    _, enc_fea_lq = vae.encode(input, return_feat=True)
    assert enc_fea_lq[1].shape == (2, 256, 256, 256)
    assert enc_fea_lq[2].shape == (2, 512, 128, 128)

    samples = torch.rand((2, 4, 64, 64))
    x_sample = vae.decode(samples, enc_fea_lq[1:3])
    assert x_sample.sample.shape == (2, 3, 512, 512)


def test_vaemodel():
    input = torch.rand((1, 3, 32, 32))
    vae = AutoencoderKL(
        act_fn='silu',
        block_out_channels=[128],
        down_block_types=['DownEncoderBlock2D'],
        in_channels=3,
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=32,
        out_channels=3,
        sample_size=128,
        up_block_types=[
            'UpDecoderBlock2D',
        ])
    output = vae.forward(input)
    assert output['sample'].shape == (1, 3, 32, 32)
