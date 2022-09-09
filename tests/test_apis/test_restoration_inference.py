# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform

import pytest
import torch

from mmedit.apis import init_model, restoration_inference


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_restoration_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + 'configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py'  # noqa
    checkpoint = None

    img_path = data_root + 'tests/data/image/lq/baboon_x4.png'

    model = init_model(config, checkpoint, device=device)

    output = restoration_inference(model, img_path)
    assert output.detach().cpu().numpy().shape == (3, 480, 500)
