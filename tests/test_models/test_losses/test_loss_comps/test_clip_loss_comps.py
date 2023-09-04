# Copyright (c) OpenMMLab. All rights reserved.
import platform

import clip
import pytest
import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmagic.models.losses import CLIPLossComps


@pytest.mark.skipif(
    'win' in platform.system().lower() and 'cu' in torch.__version__,
    reason='skip on windows-cuda due to limited RAM.')
@pytest.mark.skipif(
    digit_version(TORCH_VERSION) <= digit_version('1.6.0'),
    reason='version limitation')
def test_clip_loss():
    clip_loss_comps = CLIPLossComps(
        clip_model=dict(in_size=32),
        data_info=dict(image='fake_imgs', text='descriptions'))

    image = torch.randn(1, 3, 32, 32)
    text = 'Image for test'
    text_inputs = torch.cat([clip.tokenize(text)])
    data_dict = dict(fake_imgs=image, descriptions=text_inputs)
    loss = clip_loss_comps(outputs_dict=data_dict)
    print(loss)


def teardown_module():
    import gc
    gc.collect()
    globals().clear()
    locals().clear()
