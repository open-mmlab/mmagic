# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.stablesr.vqgan import AutoencoderKL_Resi


def test_vq_encode():
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
    assert x_sample.shape == (2, 3, 512, 512)
