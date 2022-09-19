# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.losses import FaceIdLoss


def test_face_id_loss():
    face_id_loss = FaceIdLoss(loss_weight=2.5)
    assert face_id_loss.loss_weight == 2.5
    gt, pred = torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
    loss = face_id_loss(pred, gt)
    assert loss.shape == ()
