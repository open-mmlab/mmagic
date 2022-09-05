# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from mmedit.apis import init_model, matting_inference


def test_matting_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + 'configs/dim/dim_stage3-v16-pln_1xb1-1000k_comp1k.py'
    checkpoint = None

    img_path = data_root + 'tests/data/matting_dataset/merged/GT05.jpg'
    trimap_path = data_root + 'tests/data/matting_dataset/trimap/GT05.png'

    model = init_model(config, checkpoint, device=device)

    pred_alpha = matting_inference(model, img_path, trimap_path)
    assert pred_alpha.shape == (552, 800)
