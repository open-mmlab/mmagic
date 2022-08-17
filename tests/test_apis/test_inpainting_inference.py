# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmedit.apis import inpainting_inference
from mmedit.registry import MODELS
from mmedit.utils import register_all_modules


def test_inpainting_inference():
    register_all_modules()

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')
    
    checkpoint = 'https://download.openmmlab.com/mmediting/inpainting/'\
                 'global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth'

    data_root = osp.join(osp.dirname(__file__), '../')
    config_file = osp.join(data_root, 'configs', 'gl_test.py')

    cfg = Config.fromfile(config_file)
    model = MODELS.build(cfg.model_inference)

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = cfg
    model.to(device)
    model.eval()

    masked_img_path = data_root + 'data/inpainting/celeba_test.png'
    mask_path = data_root + 'data/inpainting/bbox_mask.png'

    result = inpainting_inference(model, masked_img_path, mask_path)
    assert result.detach().cpu().numpy().shape == (3, 256, 256)
