import pytest
import torch

from mmedit.models.backbones.sr_backbones.glean_styleganv2 import \
    GLEANStyleGANv2


class TestGLEANNet:

    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(in_size=16, out_size=256, style_channels=512)

    def test_glean_styleganv2_cpu(self):
        # test default config
        glean = GLEANStyleGANv2(**self.default_cfg)
        img = torch.randn(2, 3, 16, 16)
        res = glean(img)
        assert res.shape == (2, 3, 256, 256)

        with pytest.raises(AssertionError):
            res = glean(torch.randn(2, 3, 17, 32))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='requires cuda')
    def test_glean_styleganv2_cuda(self):
        # test default config
        glean = GLEANStyleGANv2(**self.default_cfg).cuda()
        img = torch.randn(2, 3, 16, 16).cuda()
        res = glean(img)
        assert res.shape == (2, 3, 256, 256)

        with pytest.raises(AssertionError):
            res = glean(torch.randn(2, 3, 32, 17).cuda())
