# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import patch

import pytest
import torch

from mmagic.models import (PerceptualLoss, PerceptualVGG,
                           TransferalPerceptualLoss)


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
@patch.object(PerceptualVGG, 'init_weights')
def test_perceptual_loss(init_weights):
    if torch.cuda.is_available():
        loss_percep = PerceptualLoss(layer_weights={'0': 1.}).cuda()
        x = torch.randn(1, 3, 16, 16).cuda()
        x.requires_grad = True
        gt = torch.randn(1, 3, 16, 16).cuda()
        percep, style = loss_percep(x, gt)

        assert percep.item() > 0
        assert style.item() > 0

        optim = torch.optim.SGD(params=[x], lr=10)
        optim.zero_grad()
        percep.backward()
        optim.step()

        percep_new, _ = loss_percep(x, gt)
        assert percep_new < percep

        loss_percep = PerceptualLoss(
            layer_weights={
                '0': 1.
            }, perceptual_weight=0.).cuda()
        x = torch.randn(1, 3, 16, 16).cuda()
        gt = torch.randn(1, 3, 16, 16).cuda()
        percep, style = loss_percep(x, gt)
        assert percep is None and style > 0

        loss_percep = PerceptualLoss(
            layer_weights={
                '0': 1.
            }, style_weight=0., criterion='mse').cuda()
        x = torch.randn(1, 3, 16, 16).cuda()
        gt = torch.randn(1, 3, 16, 16).cuda()
        percep, style = loss_percep(x, gt)
        assert style is None and percep > 0

        loss_percep = PerceptualLoss(
            layer_weights={
                '0': 1.
            }, layer_weights_style={
                '1': 1.
            }).cuda()
        x = torch.randn(1, 3, 16, 16).cuda()
        gt = torch.randn(1, 3, 16, 16).cuda()
        percep, style = loss_percep(x, gt)
        assert percep > 0 and style > 0

    # test whether vgg type is valid
    with pytest.raises(AssertionError):
        loss_percep = PerceptualLoss(layer_weights={'0': 1.}, vgg_type='igccc')
    # test whether criterion is valid
    with pytest.raises(NotImplementedError):
        loss_percep = PerceptualLoss(
            layer_weights={'0': 1.}, criterion='igccc')

    layer_name_list = ['2', '10', '30']
    vgg_model = PerceptualVGG(
        layer_name_list,
        use_input_norm=False,
        vgg_type='vgg16',
        pretrained='torchvision://vgg16')
    x = torch.rand((1, 3, 32, 32))
    output = vgg_model(x)
    assert isinstance(output, dict)
    assert len(output) == len(layer_name_list)
    assert set(output.keys()) == set(layer_name_list)

    # test whether the layer name is valid
    with pytest.raises(AssertionError):
        layer_name_list = ['2', '10', '30', '100']
        vgg_model = PerceptualVGG(
            layer_name_list,
            use_input_norm=False,
            vgg_type='vgg16',
            pretrained='torchvision://vgg16')

    # reset mock to clear some memory usage
    init_weights.reset_mock()


def test_t_perceptual_loss():

    maps = [
        torch.rand((2, 8, 8, 8), requires_grad=True),
        torch.rand((2, 4, 16, 16), requires_grad=True)
    ]
    textures = [torch.rand((2, 8, 8, 8)), torch.rand((2, 4, 16, 16))]
    soft = torch.rand((2, 1, 8, 8))

    loss_t_percep = TransferalPerceptualLoss()
    t_percep = loss_t_percep(maps, soft, textures)
    assert t_percep.item() > 0

    loss_t_percep = TransferalPerceptualLoss(
        use_attention=False, criterion='l1')
    t_percep = loss_t_percep(maps, soft, textures)
    assert t_percep.item() > 0

    if torch.cuda.is_available():
        maps = [
            torch.rand((2, 8, 8, 8)).cuda(),
            torch.rand((2, 4, 16, 16)).cuda()
        ]
        textures = [
            torch.rand((2, 8, 8, 8)).cuda(),
            torch.rand((2, 4, 16, 16)).cuda()
        ]
        soft = torch.rand((2, 1, 8, 8)).cuda()
        loss_t_percep = TransferalPerceptualLoss().cuda()
        maps[0].requires_grad = True
        maps[1].requires_grad = True

        t_percep = loss_t_percep(maps, soft, textures)
        assert t_percep.item() > 0

        optim = torch.optim.SGD(params=maps, lr=10)
        optim.zero_grad()
        t_percep.backward()
        optim.step()

        t_percep_new = loss_t_percep(maps, soft, textures)
        assert t_percep_new < t_percep

        loss_t_percep = TransferalPerceptualLoss(
            use_attention=False, criterion='l1').cuda()
        t_percep = loss_t_percep(maps, soft, textures)
        assert t_percep.item() > 0

    # test whether vgg type is valid
    with pytest.raises(ValueError):
        TransferalPerceptualLoss(criterion='l2')


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
