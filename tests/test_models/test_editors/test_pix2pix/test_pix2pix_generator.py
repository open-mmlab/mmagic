# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch

from mmagic.models.editors.pix2pix import UnetGenerator


class TestUnetGenerator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=3,
            out_channels=3,
            num_down=8,
            base_channels=64,
            norm_cfg=dict(type='BN'),
            use_dropout=True,
            init_cfg=dict(type='normal', gain=0.02))

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_pix2pix_generator_cpu(self):
        # test with default cfg
        real_a = torch.randn((2, 3, 256, 256))
        gen = UnetGenerator(**self.default_cfg)
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['num_down'] = 7
        gen = UnetGenerator(**cfg)
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

        with pytest.raises(TypeError):
            gen = UnetGenerator(**self.default_cfg)
            gen.init_weights(pretrained=10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pix2pix_generator_cuda(self):
        # test with default cfg
        real_a = torch.randn((2, 3, 256, 256)).cuda()
        gen = UnetGenerator(**self.default_cfg).cuda()
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['num_down'] = 7
        gen = UnetGenerator(**cfg).cuda()
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

        with pytest.raises(TypeError):
            gen = UnetGenerator(**self.default_cfg)
            gen.init_weights(pretrained=10)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
