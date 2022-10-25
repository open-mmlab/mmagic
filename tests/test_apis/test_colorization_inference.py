# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch
from mmengine import Config
from mmengine.runner import load_checkpoint

from mmedit.apis import colorization_inference
from mmedit.registry import MODELS
from mmedit.utils import register_all_modules, tensor2img


def test_colorization_inference():
    register_all_modules()

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = '../../'
    config = osp.join(
        data_root,
        'configs/inst_colorization/inst-colorizatioon_cocostuff_256x256.py')

    checkpoint = None

    cfg = Config.fromfile(config)
    model = MODELS.build(cfg.model)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = cfg
    model.to(device)
    model.eval()

    img_path = '../data/image/gray/test.jpg'

    result = colorization_inference(model, img_path)
    assert tensor2img(result)[..., ::-1].shape == (256, 256, 3)
