# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors import PlainRefiner


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_plain_refiner():
    with torch.no_grad():
        plain_refiner = PlainRefiner()
        plain_refiner.init_weights()
        input = torch.rand((2, 4, 128, 128))
        raw_alpha = torch.rand((2, 1, 128, 128))
        output = plain_refiner(input, raw_alpha)
        assert output.detach().numpy().shape == (2, 1, 128, 128)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
