# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version

from mmagic.models.editors.stylegan3.stylegan3_utils import (
    apply_fractional_pseudo_rotation, apply_fractional_rotation,
    apply_fractional_translation, apply_integer_translation)


@pytest.mark.skipif(
    'win' in platform.system().lower() or not torch.cuda.is_available(),
    reason='skip due to uncompiled ops.')
def test_integer_transformation():
    x = torch.randn(1, 3, 16, 16)
    t = torch.randn(2)
    z, m = apply_integer_translation(x, t[0], t[1])
    print(z.shape)
    print(m.shape)

    # cover more lines
    t = torch.zeros(2)
    z, m = apply_integer_translation(x, t[0], t[1])

    t = torch.ones(2) * 2
    z, m = apply_integer_translation(x, t[0], t[1])


@pytest.mark.skipif(
    'win' in platform.system().lower() or not torch.cuda.is_available(),
    reason='skip due to uncompiled ops.')
def test_fractional_translation():
    x = torch.randn(1, 3, 16, 16)
    t = torch.randn(2)
    z, m = apply_fractional_translation(x, t[0], t[1])
    print(z.shape)
    print(m.shape)

    # cover more lines
    t = torch.zeros(2)
    z, m = apply_fractional_translation(x, t[0], t[1])

    t = torch.ones(2) * 2
    z, m = apply_fractional_translation(x, t[0], t[1])


@pytest.mark.skipif(
    'win' in platform.system().lower() or not torch.cuda.is_available(),
    reason='skip due to uncompiled ops.')
@pytest.mark.skipif(
    digit_version(TORCH_VERSION) < digit_version('1.8.0'),
    reason='version limitation')
def test_fractional_rotation():
    angle = torch.randn([])
    x = torch.randn(1, 3, 16, 16)
    ref, ref_mask = apply_fractional_rotation(x, angle)
    print(ref.shape)
    print(ref_mask.shape)


@pytest.mark.skipif(
    digit_version(TORCH_VERSION) < digit_version('1.8.0')
    or 'win' in platform.system().lower() or not torch.cuda.is_available(),
    reason='version limitation')
def test_fractional_pseduo_rotation():
    angle = torch.randn([])
    x = torch.randn(1, 3, 16, 16)
    ref, ref_mask = apply_fractional_pseudo_rotation(x, angle)
    print(ref.shape)
    print(ref_mask.shape)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
