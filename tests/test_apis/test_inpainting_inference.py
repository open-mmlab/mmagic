# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from mmedit.apis import init_model, inpainting_inference


def test_inpainting_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + 'configs/global_local/gl_256x256_8x12_celeba.py'
    checkpoint = 'https://download.openmmlab.com/mmediting/inpainting/ \
        global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth'

    masked_img_path = data_root + 'tests/data/inpainting/celeba_test.png'
    mask_path = data_root + 'tests/data/inpainting/bbox_mask.png'

    model = init_model(config, checkpoint, device=device)

    result = inpainting_inference(model, masked_img_path, mask_path)
    assert result.detach().cpu().numpy().shape == (3, 256, 256)
