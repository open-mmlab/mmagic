# Copyright (c) OpenMMLab. All rights reserved.
import platform

import clip
import pytest
import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.models.losses import CLIPLoss


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
@pytest.mark.skipif(
    digit_version(TORCH_VERSION) <= digit_version('1.6.0'),
    reason='version limitation')
def test_clip_loss():
    clip_loss = CLIPLoss(clip_model=dict(in_size=32, clip_type='RN50'))

    image = torch.randn(1, 3, 32, 32)
    text = 'Image for test'
    text_inputs = torch.cat([clip.tokenize(text)])
    loss = clip_loss(image, text_inputs)
    print(loss)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
