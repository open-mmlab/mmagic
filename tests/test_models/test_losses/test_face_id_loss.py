# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.losses import FaceIdLoss


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_face_id_loss():
    face_id_loss = FaceIdLoss(loss_weight=2.5)
    assert face_id_loss.loss_weight == 2.5
    gt, pred = torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
    loss = face_id_loss(pred, gt)
    assert loss.shape == ()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
