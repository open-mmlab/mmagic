# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmagic.models.editors.mspie import CatersianGrid as CSG
from mmagic.models.editors.mspie import SinusoidalPositionalEmbedding as SPE


class TestSPE:

    @classmethod
    def setup_class(cls):
        cls.spe = SPE(4, 0, 32)

    def test_spe_cpu(self):
        # test spe 1d
        embed = self.spe(torch.randn((2, 10)))
        assert embed.shape == (2, 10, 4)

        # test spe 2d
        embed = self.spe(torch.randn((2, 3, 8, 8)))
        assert embed.shape == (2, 8, 8, 8)

        with pytest.raises(AssertionError):
            _ = self.spe(torch.randn(2, 3, 3))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_spe_gpu(self):
        spe = self.spe.cuda()
        # test spe 1d
        embed = spe(torch.randn((2, 10)).cuda())
        assert embed.shape == (2, 10, 4)
        assert embed.is_cuda

        # test spe 2d
        embed = spe(torch.randn((2, 3, 8, 8)).cuda())
        assert embed.shape == (2, 8, 8, 8)

        with pytest.raises(AssertionError):
            _ = spe(torch.randn(2, 3, 3))


class TestCSG:

    @classmethod
    def setup_class(cls):
        cls.csg = CSG()

    def test_csg_cpu(self):
        csg = self.csg(torch.randn((2, 3, 4, 4)))
        assert csg.shape == (2, 2, 4, 4)

        with pytest.raises(AssertionError):
            _ = self.csg(torch.randn((2, 3, 3)))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_csg_cuda(self):
        embed = self.csg(torch.randn((2, 4, 5, 5)).cuda())
        assert embed.shape == (2, 2, 5, 5) and embed.is_cuda


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
