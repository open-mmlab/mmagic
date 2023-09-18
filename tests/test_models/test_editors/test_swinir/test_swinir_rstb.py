# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.swinir.swinir_rstb import RSTB


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_rstb():

    net = RSTB(
        dim=6, input_resolution=(8, 8), depth=6, num_heads=6, window_size=8)

    img = torch.randn(1, 64, 6)
    output = net(img, (8, 8))
    assert output.shape == (1, 64, 6)

    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda(), (8, 8))
        assert output.shape == (1, 64, 6)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
