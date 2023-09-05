# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import torch

from mmagic.models.editors.stylegan1 import get_mean_latent, style_mixing

get_module_device_str = 'mmagic.models.editors.stylegan1.stylegan_utils.get_module_device'  # noqa


@patch(get_module_device_str, MagicMock(return_value='cpu'))
def test_get_mean_latent():
    generator = MagicMock()
    generator.style_mapping = MagicMock(return_value=torch.randn(1024, 16))

    mean_style = get_mean_latent(generator)
    assert mean_style.shape == (1, 16)


def mock_generator(code, *args, **kwargs):
    if isinstance(code, torch.Tensor):
        n_sample = code.shape[0]
    elif isinstance(code, list):
        # is list
        n_sample = code[0].shape[0]
    return torch.randn(n_sample, 3, 16, 16)


@patch(get_module_device_str, MagicMock(return_value='cpu'))
def test_style_mixing():
    out = style_mixing(mock_generator, n_source=10, n_target=11)
    assert out.shape == ((1 + 10 + (1 + 10) * 11), 3, 16, 16)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
