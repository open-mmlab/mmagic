# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.runner import load_checkpoint

from mmedit.apis import inpainting_inference
from mmedit.core import tensor2img
from mmedit.models import build_model


def test_inpainting_inference():

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    checkpoint = None

    data_root = osp.join(osp.dirname(__file__), '../')
    config_file = osp.join(data_root, 'data/inpaintor_config/gl_test.py')

    cfg = mmcv.Config.fromfile(config_file)
    model = build_model(cfg.model, test_cfg=cfg.test_cfg)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = cfg
    model.to(device)
    model.eval()

    masked_img_path = data_root + 'data/image/celeba_test.png'
    mask_path = data_root + 'data/image/bbox_mask.png'

    result = inpainting_inference(model, masked_img_path, mask_path)
    result = tensor2img(result, min_max=(-1, 1))
    assert result.shape == (256, 256, 3)
