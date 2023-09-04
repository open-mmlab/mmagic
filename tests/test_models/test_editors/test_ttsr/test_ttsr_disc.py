# Copyright (c) OpenMMLab. All rights reserved.

import platform

import pytest
import torch

from mmagic.models import TTSRDiscriminator


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_ttsr_dict():
    net = TTSRDiscriminator(in_channels=3, in_size=160)
    # cpu
    inputs = torch.rand((2, 3, 160, 160))
    output = net(inputs)
    assert output.shape == (2, 1)
    # gpu
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(inputs.cuda())
        assert output.shape == (2, 1)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
