# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import unittest

import pytest
import torch
from mmengine import Config
from mmengine.runner import load_checkpoint

from mmedit.apis import colorization_inference
from mmedit.registry import MODELS
from mmedit.utils import register_all_modules, tensor2img


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_colorization_inference():
    register_all_modules()

    if not torch.cuda.is_available():
        # RoI pooling only support in GPU
        return unittest.skip('test requires GPU and torch+cuda')

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    config = osp.join(
        osp.dirname(__file__),
        '../..',
        'configs/inst_colorization/inst-colorizatioon_full_official_cocostuff-256x256.py'  # noqa
    )
    checkpoint = None

    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = cfg
    model.to(device)
    model.eval()

    img_path = osp.join(
        osp.dirname(__file__), '..', 'data/image/img_root/horse/horse.jpeg')

    result = colorization_inference(model, img_path)
    assert tensor2img(result)[..., ::-1].shape == (256, 256, 3)
