# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest.mock import patch

import pytest
import torch

from mmagic.models import IDLossModel
from mmagic.models.editors.arcface.model_irse import Backbone


def get_mock_weights():
    return Backbone(
        input_size=112, num_layers=50, drop_ratio=0.6,
        mode='ir_se').state_dict()


class TestArcFace:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            input_size=224,
            num_layers=50,
            mode='ir',
            drop_ratio=0.4,
            affine=True)

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_arcface_cpu(self):
        # test loss model
        with patch(
                'torch.hub.load_state_dict_from_url',
                return_value=get_mock_weights()):
            id_loss_model = IDLossModel()
        x1 = torch.randn((2, 3, 224, 224))
        x2 = torch.randn((2, 3, 224, 224))
        y, _ = id_loss_model(pred=x1, gt=x2)
        assert y >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_arcface_cuda(self):
        # test loss model
        with patch(
                'torch.hub.load_state_dict_from_url',
                return_value=get_mock_weights()):
            id_loss_model = IDLossModel().cuda()
        x1 = torch.randn((2, 3, 224, 224)).cuda()
        x2 = torch.randn((2, 3, 224, 224)).cuda()
        y, _ = id_loss_model(pred=x1, gt=x2)
        assert y >= 0


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
