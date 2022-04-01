# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.losses import LightCNNFeatureLoss


def test_light_cnn_feature_loss():

    pretrained = 'https://download.openmmlab.com/mmediting/' + \
        'restorers/dic/light_cnn_feature.pth'
    pred = torch.rand((3, 3, 128, 128))
    gt = torch.rand((3, 3, 128, 128))

    feature_loss = LightCNNFeatureLoss(pretrained=pretrained)
    loss = feature_loss(pred, gt)
    assert loss.item() > 0

    feature_loss = LightCNNFeatureLoss(pretrained=pretrained, criterion='mse')
    loss = feature_loss(pred, gt)
    assert loss.item() > 0

    if torch.cuda.is_available():
        pred = pred.cuda()
        gt = gt.cuda()
        feature_loss = feature_loss.cuda()
        pred.requires_grad = True

        loss = feature_loss(pred, gt)
        assert loss.item() > 0

        optim = torch.optim.SGD(params=[pred], lr=10)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_new = feature_loss(pred, gt)
        assert loss_new < loss

        feature_loss = LightCNNFeatureLoss(
            pretrained=pretrained, criterion='mse').cuda()
        loss = feature_loss(pred, gt)
        assert loss.item() > 0

    with pytest.raises(AssertionError):
        feature_loss.model.train()
        feature_loss(pred, gt)
    # test criterion value error
    with pytest.raises(ValueError):
        LightCNNFeatureLoss(pretrained=pretrained, criterion='l2')
    # test assert isinstance(pretrained, str)
    with pytest.raises(AssertionError):
        LightCNNFeatureLoss(pretrained=None)
