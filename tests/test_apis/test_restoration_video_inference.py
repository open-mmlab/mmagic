# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from mmedit.apis import init_model, restoration_video_inference


def test_restoration_video_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + '/configs/basicvsr/basicvsr_reds4.py'
    checkpoint = 'https://download.openmmlab.com/mmediting/restorers/ \
        basicvsr/basicvsr_reds4_20120409-0e599677.pth'

    input_dir = data_root + '/tests/data/frames/sequence/gt/sequence_1'

    model = init_model(config, checkpoint, device=device)

    output = restoration_video_inference(model, input_dir, 0, 0, '{:08d}.png',
                                         None)
    assert output.detach().numpy().shape == (1, 2, 3, 256, 448)
