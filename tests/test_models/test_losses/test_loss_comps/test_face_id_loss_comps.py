# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models import IDLossModel
from mmagic.models.losses import FaceIdLossComps


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
def test_face_id_loss_comps():
    face_id_loss_comps = FaceIdLossComps(
        loss_weight=2.5, data_info=dict(gt='real_imgs', pred='fake_imgs'))
    assert isinstance(face_id_loss_comps.net, IDLossModel)
    data_dict = dict(
        a=1,
        real_imgs=torch.randn(1, 3, 224, 224),
        fake_imgs=torch.randn(1, 3, 224, 224))
    loss = face_id_loss_comps(outputs_dict=data_dict)
    assert loss.shape == ()


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
