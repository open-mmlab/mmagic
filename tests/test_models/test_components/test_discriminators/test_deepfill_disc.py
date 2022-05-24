# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmedit.models.components import (DeepFillv1Discriminators,
                                      MultiLayerDiscriminator)


def test_deepfillv1_disc():
    model_config = dict(
        global_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=256,
            fc_in_channels=256 * 16 * 16,
            fc_out_channels=1,
            num_convs=4,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
        local_disc_cfg=dict(
            type='MultiLayerDiscriminator',
            in_channels=3,
            max_channels=512,
            fc_in_channels=512 * 8 * 8,
            fc_out_channels=1,
            num_convs=4,
            norm_cfg=None,
            act_cfg=dict(type='ELU'),
            out_act_cfg=dict(type='LeakyReLU', negative_slope=0.2)))
    disc = DeepFillv1Discriminators(**model_config)
    disc.init_weights()
    global_x = torch.rand((2, 3, 256, 256))
    local_x = torch.rand((2, 3, 128, 128))
    global_pred, local_pred = disc((global_x, local_x))
    assert global_pred.shape == (2, 1)
    assert local_pred.shape == (2, 1)
    assert isinstance(disc.global_disc, MultiLayerDiscriminator)
    assert isinstance(disc.local_disc, MultiLayerDiscriminator)

    with pytest.raises(TypeError):
        disc.init_weights(model_config)

    if torch.cuda.is_available():
        disc = DeepFillv1Discriminators(**model_config).cuda()
        disc.init_weights()
        global_x = torch.rand((2, 3, 256, 256)).cuda()
        local_x = torch.rand((2, 3, 128, 128)).cuda()
        global_pred, local_pred = disc((global_x, local_x))
        assert global_pred.shape == (2, 1)
        assert local_pred.shape == (2, 1)
