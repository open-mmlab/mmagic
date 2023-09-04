# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.stylegan3.stylegan3_modules import MappingNetwork


@pytest.mark.skipif(
    'win' in platform.system().lower() or not torch.cuda.is_available(),
    reason='skip on windows due to uncompiled ops.')
def test_MappingNetwork():
    mapping_network = MappingNetwork(16, 4, 5, cond_size=8)
    z = torch.randn(1, 16)
    c = torch.randn(1, 8)
    out = mapping_network(z, c)
    assert out.shape == (1, 5, 4)

    # test w/o conditional input
    mapping_network = MappingNetwork(16, 4, 5, cond_size=-1)
    out = mapping_network(z)
    assert out.shape == (1, 5, 4)

    # test w/o noise input
    mapping_network = MappingNetwork(0, 4, 5, cond_size=8)
    out = mapping_network(None, c)
    assert out.shape == (1, 5, 4)

    # test num_ws is None --> no broadcast
    mapping_network = MappingNetwork(16, 4, num_ws=None, w_avg_beta=None)
    assert not hasattr(mapping_network, 'w_avg')
    out = mapping_network(z)
    assert out.shape == (1, 4)

    # test truncation is passed
    with pytest.raises(AssertionError):
        mapping_network(z, truncation=0.9)

    mapping_network = MappingNetwork(16, 4, 5)
    out = mapping_network(z, truncation=0.9)
    assert out.shape == (1, 5, 4)

    out_trunc_work = mapping_network(z, truncation=0.9, num_truncation_layer=3)
    assert out_trunc_work.shape == (1, 5, 4)
    assert (out_trunc_work[3:] == out[3:]).all()

    # test z is None + noise_size > 0
    with pytest.raises(AssertionError):
        mapping_network(None)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
