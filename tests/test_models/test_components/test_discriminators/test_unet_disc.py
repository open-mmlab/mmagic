# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.components import UNetDiscriminatorWithSpectralNorm


def test_unet_disc_with_spectral_norm():
    # cpu
    disc = UNetDiscriminatorWithSpectralNorm(in_channels=3)
    img = torch.randn(1, 3, 16, 16)
    disc(img)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        disc.init_weights(pretrained=233)

    # cuda
    if torch.cuda.is_available():
        disc = disc.cuda()
        img = img.cuda()
        disc(img)

        with pytest.raises(TypeError):
            # pretrained must be a string path
            disc.init_weights(pretrained=233)
