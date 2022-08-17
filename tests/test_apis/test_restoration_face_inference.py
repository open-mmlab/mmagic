# # Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from mmedit.apis import init_model, restoration_face_inference


def test_restoration_face_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + 'configs/glean/glean_in128out1024_4x2_300k' + \
        '_ffhq_celebahq.py'

    checkpoint = 'https://download.openmmlab.com/mmediting/restorers/'\
        'glean/glean_in128out1024_4x2_300k_ffhq_celebahq_20210812-acbcb04f.pth'

    img_path = data_root + 'tests/data/image/face/000001.png'

    model = init_model(config, checkpoint, device=device)

    output = restoration_face_inference(model, img_path, 1, 1024)
    assert output.shape == (256, 256, 3)
