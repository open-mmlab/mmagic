# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmagic.models.editors.pggan import PGGANDiscriminator


class TestPGGANDiscriminator:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_scale=16, label_size=2)
        cls.default_inputx16 = torch.randn((2, 3, 16, 16))
        cls.default_inputx4 = torch.randn((2, 3, 4, 4))
        cls.default_inputx8 = torch.randn((2, 3, 8, 8))

    @pytest.mark.skipif(
        'win' in platform.system().lower() and 'cu' in torch.__version__,
        reason='skip on windows-cuda due to limited RAM.')
    def test_pggan_discriminator(self):
        # test with default cfg
        disc = PGGANDiscriminator(**self.default_cfg)

        score, label = disc(self.default_inputx16, transition_weight=0.1)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score, label = disc(
            self.default_inputx8, transition_weight=0.1, curr_scale=8)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score, label = disc(
            self.default_inputx4, transition_weight=0.1, curr_scale=4)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)

        disc = PGGANDiscriminator(
            in_scale=16,
            mbstd_cfg=None,
            downsample_cfg=dict(type='nearest', scale_factor=0.5))

        score = disc(self.default_inputx16, transition_weight=0.1)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score = disc(self.default_inputx8, transition_weight=0.1, curr_scale=8)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score = disc(self.default_inputx4, transition_weight=0.1, curr_scale=4)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        assert not disc.with_mbstd

        with pytest.raises(NotImplementedError):
            _ = PGGANDiscriminator(
                in_scale=16, mbstd_cfg=None, downsample_cfg=dict(type='xx'))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_pggan_discriminator_cuda(self):
        # test with default cfg
        disc = PGGANDiscriminator(**self.default_cfg).cuda()

        score, label = disc(
            self.default_inputx16.cuda(), transition_weight=0.1)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score, label = disc(
            self.default_inputx8.cuda(), transition_weight=0.1, curr_scale=8)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)
        score, label = disc(
            self.default_inputx4.cuda(), transition_weight=0.1, curr_scale=4)
        assert score.shape == (2, 1)
        assert label.shape == (2, 2)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
