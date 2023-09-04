# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmagic.models.editors.ddpm.denoising_unet import (DenoisingUnet,
                                                       NormWithEmbedding)


def test_DenoisingUnet():
    input = torch.rand((1, 3, 32, 32))
    unet = DenoisingUnet(32)
    output = unet.forward(input, 10)
    assert output['sample'].shape == (1, 6, 32, 32)


def test_NormWithEmbedding():
    input = torch.rand((4, 32))
    emb = torch.rand((4, 32))
    ins = NormWithEmbedding(32, 32)
    output = ins.forward(input, emb)
    assert output.shape == (4, 32, 4, 32)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
