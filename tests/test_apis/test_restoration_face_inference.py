# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import pytest
import torch

from mmedit import digit_version
from mmedit.apis import init_model, restoration_face_inference


@pytest.mark.skipif(
    digit_version(torch.__version__) < (1, 6),
    reason='requires torch-1.6.0 or higher')
def test_restoration_face_inference():

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    checkpoint = None

    data_root = osp.join(osp.dirname(__file__), '../../')
    config = data_root + 'configs/restorers/glean/glean_in128out1024_4x2_300k_ffhq_celebahq.py'  # noqa
    img_path = data_root + 'tests/data/face/000001.png'

    model = init_model(config, checkpoint, device=device)

    output = restoration_face_inference(model, img_path, 1, 1024)
    assert output.shape == (256, 256, 3)
