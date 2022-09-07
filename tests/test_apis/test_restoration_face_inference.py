# # Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys

import pytest
import torch

from mmedit.apis import init_model, restoration_face_inference


@pytest.mark.skipif(
    sys.platform == 'win32' and torch.cuda.is_available(),
    reason='skip on windows-cuda due to limited RAM.')
def test_restoration_face_inference():
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + 'configs/glean/glean_in128out1024_4xb2-300k_ffhq-celeba-hq.py'  # noqa

    checkpoint = None

    img_path = data_root + 'tests/data/image/face/000001.png'

    model = init_model(config, checkpoint, device=device)

    output = restoration_face_inference(model, img_path, 1, 1024)
    assert output.shape == (256, 256, 3)
