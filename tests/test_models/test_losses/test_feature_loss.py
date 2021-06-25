import pytest
import torch

from mmedit.models.losses import FeatureLoss


def test_feature_loss():
    pretrained = 'https://download.openmmlab.com/mmediting/' + \
        'restorers/dic/light_cnn_feature.pth'
    pred = torch.rand((3, 3, 128, 128))
    gt = torch.rand((3, 3, 128, 128))

    feature_loss = FeatureLoss(pretrained=pretrained)
    loss = feature_loss(pred, gt)
    assert loss.item() > 0

    feature_loss = FeatureLoss(pretrained=pretrained, criterion='mse')
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

        feature_loss = FeatureLoss(
            pretrained=pretrained, criterion='mse').cuda()
        loss = feature_loss(pred, gt)
        assert loss.item() > 0

    # test criterion value error
    with pytest.raises(ValueError):
        FeatureLoss(pretrained=pretrained, criterion='l2')
    # test assert isinstance(pretrained, str)
    with pytest.raises(AssertionError):
        FeatureLoss(pretrained=None)


if __name__ == '__main__':
    test_feature_loss()
