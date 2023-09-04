# Copyright (c) OpenMMLab. All rights reserved.
import platform
from unittest import TestCase

import pytest
import torch
from mmengine import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.models.editors.stylegan2.ada.augment import AugmentPipe


class TestAuementPipe(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.default_cfg = dict(
            xflip=1,
            rotate90=1,
            xint=1,
            xint_max=1,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            scale_std=1,
            rotate_max=1,
            aniso_std=1,
            xfrac_std=1,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
            brightness_std=1,
            contrast_std=1,
            hue_max=1,
            saturation_std=1,
            imgfilter=1,
            imgfilter_bands=[1, 1, 1, 1],
            imgfilter_std=1,
            noise=1,
            cutout=1,
            noise_std=1,
            cutout_size=0.5,
        )

    @pytest.mark.skipif(
        digit_version(TORCH_VERSION) <= digit_version('1.6.0')
        or 'win' in platform.system().lower() or not torch.cuda.is_available(),
        reason=('torch version lower than 1.7.0 does not have '
                '`torch.exp2` api, skip on windows due to uncompiled ops.'))
    def test_forward(self):
        augment_pipeline = AugmentPipe(**self.default_cfg)

        inp = torch.rand(2, 3, 64, 64)
        out = augment_pipeline(inp)
        assert out.shape == (2, 3, 64, 64)

        out = augment_pipeline(inp, debug_percentile=0.1)

        no_aug_cfg = {
            k: 0
            for k, v in self.default_cfg.items()
            if isinstance(v, (float, int))
        }
        augment_pipeline = AugmentPipe(**no_aug_cfg)

        inp = torch.rand(2, 3, 64, 64)
        out = augment_pipeline(inp)
        assert out.shape == (2, 3, 64, 64)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
