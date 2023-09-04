# Copyright (c) OpenMMLab. All rights reserved.
import platform
from copy import deepcopy

import pytest
import torch

from mmagic.models.editors.cyclegan import ResnetGenerator


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
class TestResnetGenerator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            in_channels=3,
            out_channels=3,
            base_channels=64,
            norm_cfg=dict(type='IN'),
            use_dropout=False,
            num_blocks=9,
            padding_mode='reflect',
            init_cfg=dict(type='normal', gain=0.02))

    def test_cyclegan_generator_cpu(self):
        # test with default cfg
        real_a = torch.randn((2, 3, 256, 256))
        gen = ResnetGenerator(**self.default_cfg)
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['num_blocks'] = 8
        gen = ResnetGenerator(**cfg)
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_cyclegan_generator_cuda(self):
        # test with default cfg
        real_a = torch.randn((2, 3, 256, 256)).cuda()
        gen = ResnetGenerator(**self.default_cfg).cuda()
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)

        # test args system
        cfg = deepcopy(self.default_cfg)
        cfg['num_blocks'] = 8
        gen = ResnetGenerator(**cfg).cuda()
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
