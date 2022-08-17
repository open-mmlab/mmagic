# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models import UNetDiscriminatorWithSpectralNorm


def test_unet_disc_with_spectral_norm():
    # cpu
    disc = UNetDiscriminatorWithSpectralNorm(in_channels=3)
    img = torch.randn(1, 3, 16, 16)
    output = disc(img)
    assert output.detach().numpy().shape == (1, 1, 16, 16)

    # cuda
    if torch.cuda.is_available():
        disc = disc.cuda()
        img = img.cuda()
        output = disc(img)
        assert output.detach().cpu().numpy().shape == (1, 1, 16, 16)
