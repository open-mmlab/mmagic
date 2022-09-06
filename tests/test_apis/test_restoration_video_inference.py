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
    config = osp.join(data_root, '/configs/basicvsr/basicvsr_2xb4_reds4.py')
    checkpoint = None

    input_dir = osp.join(data_root,
                         '/tests/data/frames/sequence/gt/sequence_1')

    model = init_model(config, checkpoint, device=device)

    output = restoration_video_inference(model, input_dir, 0, 0, '{:08d}.png',
                                         None)
    assert output.detach().numpy().shape == (1, 2, 3, 256, 448)
