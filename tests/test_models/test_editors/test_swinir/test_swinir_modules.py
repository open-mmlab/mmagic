# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.swinir.swinir_modules import (PatchEmbed,
                                                         PatchUnEmbed,
                                                         Upsample,
                                                         UpsampleOneStep)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_patchEmbed():

    net = PatchEmbed(
        img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None)

    img = torch.randn(1, 3, 4, 4)
    output = net(img)
    assert output.shape == (1, 16, 3)

    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 16, 3)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_patchUnEmbed():

    net = PatchUnEmbed(
        img_size=16, patch_size=4, in_chans=3, embed_dim=3, norm_layer=None)

    img = torch.randn(1, 64, 3)
    output = net(img, (8, 8))
    assert output.shape == (1, 3, 8, 8)

    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda(), (8, 8))
        assert output.shape == (1, 3, 8, 8)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_upsample():

    net = Upsample(scale=2, num_feat=3)

    img = torch.randn(1, 3, 8, 8)
    output = net(img)
    assert output.shape == (1, 3, 16, 16)

    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 16, 16)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_upsampleOneStep():

    net = UpsampleOneStep(
        scale=2,
        num_feat=3,
        num_out_ch=4,
    )

    img = torch.randn(1, 3, 8, 8)
    output = net(img)
    assert output.shape == (1, 4, 16, 16)

    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 4, 16, 16)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
