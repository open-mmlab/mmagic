# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.ddpm.denoising_unet import DenoisingUnet


def test_DenoisingUnet():
    input = torch.rand((1, 3, 32, 32))
    unet = DenoisingUnet(32)
    output = unet.forward(input, 10)
    assert output['outputs'].shape == (1, 6, 32, 32)


if __name__ == '__main__':
    test_DenoisingUnet()
