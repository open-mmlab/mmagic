# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from mmedit.apis import init_model, restoration_inference


def test_restoration_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + 'configs/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py'
    checkpoint = 'https://download.openmmlab.com/mmediting/restorers/ \
        basicvsr/basicvsr_reds4_20120409-0e599677.pth'

    img_path = data_root + 'tests/data/image/lq/baboon_x4.png'

    model = init_model(config, checkpoint, device=device)

    output = restoration_inference(model, img_path)
    assert output.detach().cpu().numpy().shape == (3, 480, 500)
