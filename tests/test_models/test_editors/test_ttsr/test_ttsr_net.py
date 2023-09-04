# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.ttsr.ttsr_net import (CSFI2, CSFI3, SFE,
                                                 MergeFeatures)
from mmagic.registry import MODELS


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_sfe():
    inputs = torch.rand(2, 3, 48, 48)
    sfe = SFE(3, 64, 16, 1.)
    outputs = sfe(inputs)
    assert outputs.shape == (2, 64, 48, 48)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_csfi():
    inputs1 = torch.rand(2, 16, 24, 24)
    inputs2 = torch.rand(2, 16, 48, 48)
    inputs4 = torch.rand(2, 16, 96, 96)

    csfi2 = CSFI2(mid_channels=16)
    out1, out2 = csfi2(inputs1, inputs2)
    assert out1.shape == (2, 16, 24, 24)
    assert out2.shape == (2, 16, 48, 48)

    csfi3 = CSFI3(mid_channels=16)
    out1, out2, out4 = csfi3(inputs1, inputs2, inputs4)
    assert out1.shape == (2, 16, 24, 24)
    assert out2.shape == (2, 16, 48, 48)
    assert out4.shape == (2, 16, 96, 96)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_merge_features():
    inputs1 = torch.rand(2, 16, 24, 24)
    inputs2 = torch.rand(2, 16, 48, 48)
    inputs4 = torch.rand(2, 16, 96, 96)

    merge_features = MergeFeatures(mid_channels=16, out_channels=3)
    out = merge_features(inputs1, inputs2, inputs4)
    assert out.shape == (2, 3, 96, 96)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_ttsr_net():
    inputs = torch.rand(2, 3, 24, 24)
    soft_attention = torch.rand(2, 1, 24, 24)
    t_level3 = torch.rand(2, 64, 24, 24)
    t_level2 = torch.rand(2, 32, 48, 48)
    t_level1 = torch.rand(2, 16, 96, 96)

    ttsr_cfg = dict(
        type='TTSRNet',
        in_channels=3,
        out_channels=3,
        mid_channels=16,
        texture_channels=16)
    ttsr = MODELS.build(ttsr_cfg)
    outputs = ttsr(inputs, soft_attention, (t_level3, t_level2, t_level1))

    assert outputs.shape == (2, 3, 96, 96)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
