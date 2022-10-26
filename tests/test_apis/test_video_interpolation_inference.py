# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch

from mmedit.apis import init_model, video_interpolation_inference


def test_video_interpolation_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + 'configs/cain/cain_g1b32_1xb5_vimeo90k-triplet.py'
    checkpoint = None

    input_dir = data_root + 'tests/data/frames/test_inference.mp4'

    model = init_model(config, checkpoint, device=device)

    video_interpolation_inference(
        model=model, input_dir=input_dir, output_dir='out', fps=60.0)


test_video_interpolation_inference()
