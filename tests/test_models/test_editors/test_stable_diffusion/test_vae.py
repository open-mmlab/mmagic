# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.editors.stable_diffusion.vae import AutoencoderKL


def test_vae():
    input = torch.rand((1, 3, 32, 32))
    vae = AutoencoderKL()
    output = vae.forward(input)
    assert output['sample'].shape == (1, 3, 32, 32)


if __name__ == '__main__':
    test_vae()
